//! Supervised training of the Inkling 2 text decoder on the XLA backend.
//!
//! An [`Inkling2Trainer`] owns everything that one training run keeps on the device, namely the parameters of an
//! [`Inkling2Decoder`] and the state of an [`AdamW`] optimizer, together with one compiled function that performs a
//! complete training step. The step computes the loss of [`Inkling2Decoder::loss`] and its gradients with respect to
//! every parameter by reverse mode differentiation, turns the gradients into updates with the optimizer, and applies
//! them. All of that is one program, so a step is a single execution on the device with no host round trip other than
//! reading back the loss. The inputs of a step are the parameters, the optimizer state, and an [`Inkling2Batch`], and
//! its outputs are the loss and the next parameters and optimizer state, which replace the previous ones.
//!
//! The expert projections, which hold almost all parameters, are computed either at the floating-point compute data
//! type of the decoder or in the block-scaled formats of the Inkling 2 releases with their mixed precision backward
//! pass, as [`Inkling2ExpertPrecision`] selects. The parameters themselves are always `f32` master values.
//!
//! The first step compiles the program for the shapes of its batch, so its duration is dominated by compilation.
//! Later steps with the same batch shapes reuse the compiled program. [`Inkling2Trainer::step`] reports the two
//! separately through [`Inkling2StepReport`].
//!
//! # Checkpoints
//!
//! The training state of a run (i.e., the parameters and the optimizer state) is saved and loaded through a
//! [`training::Checkpointer`] whose [`Context`] [`Inkling2Trainer::checkpoint_context`] creates for an array format.
//! [`Inkling2Trainer::save`] offers the current step to the checkpointer, whose policies decide whether it is saved
//! and which older checkpoints are kept, and [`Inkling2Trainer::load`] continues a run from a saved step. Leaves
//! move between the device and storage one at a time through an [`XlaLeafHandler`], so neither direction holds the
//! whole state in host memory.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use ryft_checkpoints::{ArrayFormat, CheckpointError, Context, LeafHandler, SaveOptions, XlaLeafHandler, training};
use ryft_core::{
    Array as HostArray, ArrayType, DataType, Differentiate, MeshAxis, MeshAxisType, Parameter, Parameterized,
    ParameterizedFamily, ProgramError, ProjectedContext, RandomAlgorithm, Value,
};
use ryft_macros::Parameterized as ParameterizedDerive;
use ryft_models::experts::{BlockScaledExpertProjector, BlockScaledProjections, FloatingExpertProjector};
use ryft_models::inkling2::{Inkling2Configuration, Inkling2ConfigurationError, Inkling2Decoder, Inkling2Inputs};
use ryft_optimizers::{AdamW, AdamWState, Optimizer, OptimizerError, UpdateArguments, apply_updates};
use ryft_xla::{Array, CompileTracer, Error as XlaError, Options, Session, jit_with_options};
use thiserror::Error;

// TODO(eaplatanios): Review this module.

/// Error returned by the Inkling 2 training experiment.
#[derive(Debug, Error)]
pub enum Inkling2TrainingError {
    #[error(transparent)]
    Xla(#[from] XlaError),

    #[error(transparent)]
    Program(#[from] ProgramError),

    #[error(transparent)]
    Checkpoint(#[from] CheckpointError),

    #[error(transparent)]
    Optimizer(#[from] OptimizerError),

    #[error(transparent)]
    Configuration(#[from] Inkling2ConfigurationError),

    #[error("invalid Inkling 2 training batch: {message}")]
    InvalidBatch { message: String },
}

/// Numerical precision of the expert projections of a training run.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Inkling2ExpertPrecision {
    /// Expert projections are floating-point contractions at the compute data type of the decoder, with ordinary
    /// gradients (refer to [`FloatingExpertProjector`]).
    Floating,

    /// Expert projections follow the mixed precision recipe of the Inkling 2 models. The forward pass contracts
    /// NVFP4 values in the leading layers and MXFP8 values in the remaining ones, and the backward pass contracts
    /// MXFP8 values that are quantized again with stochastic rounding (refer to [`BlockScaledExpertProjector`]). The
    /// number of tokens of a batch must be a multiple of 32, and the backend must support the narrow element types of
    /// both formats.
    BlockScaled,
}

/// One training batch of the Inkling 2 decoder, which pairs the inputs of the decoder with the next token labels and
/// the supervision weights of every position.
#[derive(Clone, Debug, PartialEq, Eq, Hash, ParameterizedDerive)]
#[ryft(crate = "ryft_core")]
pub struct Inkling2Batch<P: Parameter> {
    /// Inputs of the decoder.
    inputs: Inkling2Inputs<P>,

    /// Next token identifiers, as an `i32` array of shape `[batch, sequence]`.
    labels: P,

    /// Supervision weights, as an `f32` array of shape `[batch, sequence]`.
    weights: P,
}

impl<P: Parameter> Inkling2Batch<P> {
    /// Creates a new [`Inkling2Batch`].
    #[inline]
    pub fn new(inputs: Inkling2Inputs<P>, labels: P, weights: P) -> Self {
        Self { inputs, labels, weights }
    }

    /// Returns the inputs of the decoder.
    #[inline]
    pub fn inputs(&self) -> &Inkling2Inputs<P> {
        &self.inputs
    }

    /// Returns the next token identifiers.
    #[inline]
    pub fn labels(&self) -> &P {
        &self.labels
    }

    /// Returns the supervision weights.
    #[inline]
    pub fn weights(&self) -> &P {
        &self.weights
    }
}

impl Inkling2Batch<HostArray> {
    /// Creates a batch of `batch_size` sequences of `sequence_length` reproducible pseudo-random tokens, each of
    /// which is one document. Every position except the last one of a sequence is supervised with the token that
    /// follows it, so a model can only lower the loss of this batch by memorizing it, which is what a smoke test of a
    /// training step needs.
    ///
    /// # Errors
    ///
    /// Returns [`Inkling2TrainingError::InvalidBatch`] if the batch would be empty or would have no supervised
    /// position.
    pub fn synthetic(
        configuration: &Inkling2Configuration,
        batch_size: usize,
        sequence_length: usize,
        seed: u64,
    ) -> Result<Self, Inkling2TrainingError> {
        if batch_size == 0 || sequence_length < 2 {
            return Err(Inkling2TrainingError::InvalidBatch {
                message: format!(
                    "a synthetic batch needs at least one sequence of at least two tokens but got {batch_size} \
                     sequences of {sequence_length} tokens",
                ),
            });
        }

        // Tokens come from a SplitMix64 generator, reduced to the usable vocabulary.
        let mut state = seed;
        let mut next_token = || {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut output = state;
            output = (output ^ (output >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            output = (output ^ (output >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            ((output ^ (output >> 31)) % configuration.usable_vocabulary_size() as u64) as i32
        };
        let count = batch_size * sequence_length;
        let tokens = (0..count).map(|_| next_token()).collect::<Vec<_>>();
        let positions = (0..count).map(|index| (index % sequence_length) as i32).collect::<Vec<_>>();
        let labels = (0..count)
            .map(|index| if (index + 1) % sequence_length == 0 { 0 } else { tokens[index + 1] })
            .collect::<Vec<_>>();
        let weights = (0..count)
            .map(|index| if (index + 1) % sequence_length == 0 { 0.0f32 } else { 1.0 })
            .collect::<Vec<_>>();
        let shape = [batch_size, sequence_length];
        let integers = |values: &[i32]| HostArray::from_elements(ArrayType::new_static(DataType::I32, shape), values);
        Ok(Self {
            inputs: Inkling2Inputs::new(
                integers(&tokens)?,
                integers(&positions)?,
                integers(&vec![0; count])?,
                HostArray::from_elements(ArrayType::new_static(DataType::Boolean, shape), &vec![true; count])?,
            ),
            labels: integers(&labels)?,
            weights: HostArray::from_elements(ArrayType::new_static(DataType::F32, shape), &weights)?,
        })
    }
}

/// Measurements of one training step.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Inkling2StepReport {
    /// Loss of the batch before the update of the step.
    loss: f32,

    /// Wall-clock duration of the step in seconds, from handing the batch to the device until the loss is read
    /// back, which waits for the whole step to finish.
    seconds: f64,

    /// Whether the step compiled its program (i.e., whether it was the first step for the shapes of its batch).
    compiled: bool,
}

impl Inkling2StepReport {
    /// Returns the loss of the batch before the update of the step.
    #[inline]
    pub fn loss(&self) -> f32 {
        self.loss
    }

    /// Returns the wall-clock duration of the step in seconds, which includes reading the loss back.
    #[inline]
    pub fn seconds(&self) -> f64 {
        self.seconds
    }

    /// Returns whether the step compiled its program.
    #[inline]
    pub fn compiled(&self) -> bool {
        self.compiled
    }
}

/// State of the [`AdamW`] optimizer of an [`Inkling2Trainer`], whose per-leaf statistics have the structure of the
/// decoder, so the paths of their leaves follow those of the parameters (e.g., `$.adam.first_moments.embedding`).
pub type Inkling2OptimizerState<P> = AdamWState<P, Inkling2Decoder<P>>;

/// Inputs of the compiled training step, which end with the random generator state of the step.
type StepInputs<P> = (Inkling2Decoder<P>, Inkling2OptimizerState<P>, Inkling2Batch<P>, P);

/// Outputs of the compiled training step.
type StepOutputs<P> = (P, Inkling2Decoder<P>, Inkling2OptimizerState<P>);

/// Training state that a checkpoint of an [`Inkling2Trainer`] holds, which consists of the parameters of the decoder
/// and the state of the optimizer. The number of steps that the trainer has run is the step of the checkpoint.
pub type Inkling2TrainingState<P> = (Inkling2Decoder<P>, Inkling2OptimizerState<P>);

/// Supervised training run of an Inkling 2 decoder on one device of an XLA [`Session`]. Refer to the
/// [module documentation](self) for what a step does.
pub struct Inkling2Trainer<'o> {
    /// Session that executes the training step.
    session: &'o Session,

    /// Handler that moves leaves between the host and the single device mesh on which the parameters live.
    leaf_handler: XlaLeafHandler<'o>,

    /// Compiled training step.
    step: Box<dyn 'o + Fn(StepInputs<Array>) -> Result<StepOutputs<Array>, XlaError>>,

    /// Parameters of the decoder and state of the optimizer, which are [`None`] only while a step is running or
    /// after a step failed.
    state: Option<Inkling2TrainingState<Array>>,

    /// Seed of the random generator states of the steps.
    seed: u64,

    /// Number of steps that this trainer has run.
    step_count: usize,
}

impl<'o> Inkling2Trainer<'o> {
    /// Creates a trainer that moves `decoder` to the first device of `session` and trains it with `optimizer`.
    ///
    /// # Parameters
    ///
    ///   - `session`: XLA session whose platform executes the training step.
    ///   - `configuration`: Configuration of the decoder.
    ///   - `decoder`: Initial parameters of the decoder on the host, which are uploaded leaf by leaf.
    ///   - `optimizer`: Optimizer that turns gradients into updates.
    ///   - `expert_precision`: Numerical precision of the expert projections.
    ///   - `seed`: Seed of the random generator states of the steps, which seed stochastic rounding.
    ///
    /// # Errors
    ///
    /// Returns an [`Inkling2TrainingError`] if the configuration is invalid, if a parameter cannot be uploaded, or if
    /// the optimizer cannot initialize its state.
    pub fn new(
        session: &'o Session,
        configuration: Inkling2Configuration,
        decoder: Inkling2Decoder<HostArray>,
        optimizer: AdamW<f64>,
        expert_precision: Inkling2ExpertPrecision,
        seed: u64,
    ) -> Result<Self, Inkling2TrainingError> {
        configuration.validate()?;
        let mesh = session.mesh([MeshAxis::new("device", 1, MeshAxisType::Auto).map_err(XlaError::from)?])?;
        let leaf_handler = XlaLeafHandler::new(session, mesh.clone());
        // The optimizer state is initialized on the host and uploaded with the parameters, so that constructing a
        // trainer executes nothing on the device.
        let optimizer_state: Inkling2OptimizerState<HostArray> = optimizer.initialize(&decoder)?;
        let optimizer_state = upload(&leaf_handler, optimizer_state)?;
        let decoder = upload(&leaf_handler, decoder)?;

        // The parameters and the optimizer state of a step are replaced by its outputs, so their device buffers are
        // donated to the step, which lets the device update them in place.
        let donation = (
            decoder.parameter_structure().map_parameters(|_| true).map_err(XlaError::from)?,
            optimizer_state.parameter_structure().map_parameters(|_| true).map_err(XlaError::from)?,
            Inkling2Batch::new(Inkling2Inputs::new(false, false, false, false), false, false),
            false,
        );
        let projections = BlockScaledProjections::<ProjectedContext<Session, ArrayType>>::new()?;
        let dispatcher = jit_with_options::<_, (), StepInputs<ArrayType>, StepOutputs<ArrayType>>(
            move |(), (decoder, optimizer_state, batch, random_state): StepInputs<CompileTracer>| {
                let domain = batch.labels.domain();
                let compute_type = configuration.compute_type();
                let (loss, gradients) =
                    domain.differentiate_at(decoder.clone()).with_captures((batch, random_state)).value_and_gradient(
                        |decoder, (batch, random_state): (Inkling2Batch<_>, _)| match expert_precision {
                            Inkling2ExpertPrecision::Floating => decoder.loss(
                                &configuration,
                                &FloatingExpertProjector::new(compute_type),
                                &batch.inputs,
                                &batch.labels,
                                &batch.weights,
                            ),
                            Inkling2ExpertPrecision::BlockScaled => decoder.loss(
                                &configuration,
                                &BlockScaledExpertProjector::new(
                                    &projections,
                                    configuration.nvfp4_layer_count(),
                                    compute_type,
                                    random_state,
                                ),
                                &batch.inputs,
                                &batch.labels,
                                &batch.weights,
                            ),
                        },
                    )?;
                let (updates, optimizer_state) = optimizer
                    .update(gradients, optimizer_state, &decoder, &UpdateArguments::new())
                    .map_err(training_step_error)?;
                let decoder = apply_updates(decoder, updates).map_err(training_step_error)?;
                Ok((loss, decoder, optimizer_state))
            },
            session,
            Options::new(mesh.clone()).with_donate(donation),
        );
        Ok(Self {
            session,
            leaf_handler,
            step: Box::new(move |inputs| dispatcher.execute((), inputs)),
            state: Some((decoder, optimizer_state)),
            seed,
            step_count: 0,
        })
    }

    /// Returns the session that executes the training step.
    #[inline]
    pub fn session(&self) -> &'o Session {
        self.session
    }

    /// Returns the number of steps that this trainer has run.
    #[inline]
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Returns the current parameters of the decoder on the device, or [`None`] after a step failed, because a
    /// failed step leaves the donated parameters in an unspecified state.
    #[inline]
    pub fn decoder(&self) -> Option<&Inkling2Decoder<Array>> {
        self.state.as_ref().map(|(decoder, _)| decoder)
    }

    /// Returns a [`Context`] for the checkpoints of this trainer that stores arrays in `array_format`, which reads the
    /// leaves of the training state back from their device when it saves them and loads them directly into device
    /// buffers. A [`training::Checkpointer`] for [`Self::save`] and [`Self::load`] is created with it.
    #[inline]
    pub fn checkpoint_context(&self, array_format: Arc<dyn ArrayFormat>) -> Context<XlaLeafHandler<'o>> {
        Context::new(XlaLeafHandler::new(self.session, self.leaf_handler.mesh().clone()), array_format)
    }

    /// Saves the training state as the checkpoint of the current step through `checkpointer`, whose policies decide
    /// whether the step is saved and which older checkpoints are kept. The leaves are read back from the device one
    /// at a time, and only when the step is saved. Returns whether the step was saved.
    ///
    /// # Parameters
    ///
    ///   - `checkpointer`: Checkpointer of this training run, created with [`Self::checkpoint_context`].
    ///   - `loss`: Loss to record as the `loss` metric of the checkpoint, if any (e.g., that of the latest step).
    ///
    /// # Errors
    ///
    /// Returns an [`Inkling2TrainingError`] if the trainer has no parameters because an earlier step failed, or if
    /// the checkpoint cannot be saved.
    pub fn save(
        &self,
        checkpointer: &mut training::Checkpointer<XlaLeafHandler<'o>>,
        loss: Option<f32>,
    ) -> Result<bool, Inkling2TrainingError> {
        let Some(state) = &self.state else {
            return Err(Inkling2TrainingError::InvalidBatch {
                message: "the trainer has no parameters because an earlier step failed".to_string(),
            });
        };
        let options = match loss {
            Some(loss) => SaveOptions::new().with_metrics(BTreeMap::from([("loss".to_string(), f64::from(loss))])),
            None => SaveOptions::new(),
        };
        Ok(checkpointer.save(self.step_count as u64, state, &options)?)
    }

    /// Replaces the training state of this trainer with the checkpoint of `step` of `checkpointer`, or with its latest
    /// checkpoint when `step` is [`None`], and continues counting steps from that step, so that the following steps
    /// draw the random generator states that they drew when they first ran. The leaves are loaded directly into
    /// device buffers, one at a time. Returns the step of the loaded checkpoint.
    ///
    /// # Errors
    ///
    /// Returns an [`Inkling2TrainingError`] if the trainer has no parameters to take the structure and the leaf types
    /// of the state from, because an earlier step failed, or if the checkpoint does not exist or does not hold a
    /// state of that structure.
    pub fn load(
        &mut self,
        checkpointer: &mut training::Checkpointer<XlaLeafHandler<'o>>,
        step: Option<u64>,
    ) -> Result<u64, Inkling2TrainingError> {
        let Some(state) = &self.state else {
            return Err(Inkling2TrainingError::InvalidBatch {
                message: "the trainer has no parameters because an earlier step failed".to_string(),
            });
        };
        let step = match step.or(checkpointer.latest().map(training::CheckpointMetadata::step)) {
            Some(step) => step,
            None => {
                return Err(CheckpointError::MissingCheckpoint {
                    message: format!("`{}` holds no checkpoint", checkpointer.directory()),
                }
                .into());
            }
        };
        let abstract_state = state
            .clone()
            .map_parameters(|leaf| ArrayType::new(leaf.data_type(), leaf.shape().into()))
            .map_err(CheckpointError::from)?;
        self.state = Some(checkpointer.load(Some(step), abstract_state)?);
        self.step_count = step as usize;
        Ok(step)
    }

    /// Runs one training step on `batch` and returns its [`Inkling2StepReport`]. The reported duration covers
    /// uploading the batch, executing the step, and reading back the loss, which waits for the step to finish.
    ///
    /// # Errors
    ///
    /// Returns an [`Inkling2TrainingError`] if the batch cannot be uploaded, if the step fails to compile or execute,
    /// or if an earlier step failed.
    pub fn step(&mut self, batch: &Inkling2Batch<HostArray>) -> Result<Inkling2StepReport, Inkling2TrainingError> {
        let Some((decoder, optimizer_state)) = self.state.take() else {
            return Err(Inkling2TrainingError::InvalidBatch {
                message: "the trainer has no parameters because an earlier step failed".to_string(),
            });
        };
        let start = Instant::now();
        let batch = upload(&self.leaf_handler, batch.clone())?;

        // Every step gets its own random generator state, which is derived from the seed and the step count.
        let random_state =
            HostArray::from_elements(RandomAlgorithm::ThreeFry.state_type(), &[self.seed, self.step_count as u64])?;
        let random_state = upload(&self.leaf_handler, random_state)?;
        let (loss, decoder, optimizer_state) = (self.step)((decoder, optimizer_state, batch, random_state))?;
        let loss = loss.copy_to_host_bytes()?;
        let seconds = start.elapsed().as_secs_f64();
        let loss = f32::from_ne_bytes(loss.as_slice().try_into().map_err(|_| Inkling2TrainingError::InvalidBatch {
            message: "the training step did not return a rank-0 `f32` loss".to_string(),
        })?);
        self.state = Some((decoder, optimizer_state));
        self.step_count += 1;
        Ok(Inkling2StepReport { loss, seconds, compiled: self.step_count == 1 })
    }
}

/// Uploads the leaves of `structure` to the device mesh of `leaf_handler`, one at a time.
fn upload<T: Parameterized<HostArray>>(
    leaf_handler: &XlaLeafHandler<'_>,
    structure: T,
) -> Result<T::To<Array>, CheckpointError>
where
    T::Family: ParameterizedFamily<Array>,
{
    structure.try_map_named_parameters(|path, leaf| leaf_handler.deserialize(path, leaf))
}

/// Converts an optimizer error that occurred while tracing the training step into an XLA error.
fn training_step_error(error: OptimizerError) -> XlaError {
    XlaError::from(ProgramError::InvalidArgument { message: error.to_string() })
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use ryft_xla::Platform;

    #[cfg(feature = "safetensors")]
    use ryft_checkpoints::SafetensorsFormat;
    #[cfg(feature = "safetensors")]
    use ryft_checkpoints::training::{Checkpointer, CheckpointerOptions, FixedIntervalPolicy, LatestN};

    use super::*;

    /// Returns a configuration with 8 layers and a loop over layers `2..6`, which is small enough to compile quickly.
    fn test_configuration() -> Inkling2Configuration {
        let json = r#"{"text_config": {
            "hidden_size": 32, "num_hidden_layers": 8, "vocab_size": 64, "unpadded_vocab_size": 60,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8, "swa_num_attention_heads": 4,
            "swa_num_key_value_heads": 2, "swa_head_dim": 8, "d_rel": 4, "rel_extent": 6, "sliding_window_size": 4,
            "local_layer_ids": [0, 2, 4, 6], "log_scaling_n_floor": 4, "log_scaling_alpha": 0.1,
            "rms_norm_eps": 1e-06, "qk_l2_clip_threshold": 2.5, "rel_l2_clip_threshold": 1.5,
            "sconv_kernel_size": 4, "final_logit_softcapping": 30.0, "n_layers_per_attn_res_block": 2,
            "n_routed_experts": 4, "num_experts_per_tok": 2, "intermediate_size": 32, "mlp_gate_softcap": 4.0,
            "mlp_up_softcap": 25.0, "route_scale": 8.0,
            "loop_config": {"core_start": 2, "core_end": 6, "num_passes": 2, "gate_rank": 8}
        }}"#;
        Inkling2Configuration::from_json(json)
            .unwrap()
            .with_nvfp4_layer_count(6)
            .with_compute_type(DataType::F32)
    }

    /// Trains the test decoder for `steps` steps on one synthetic batch and returns the reports of the steps.
    fn train(expert_precision: Inkling2ExpertPrecision, steps: usize) -> Vec<Inkling2StepReport> {
        let configuration = test_configuration();
        let session = Session::builder().platform(Platform::Cpu).build().unwrap();
        let decoder = Inkling2Decoder::random(&configuration, 1).unwrap();
        let batch = Inkling2Batch::synthetic(&configuration, 1, 32, 2).unwrap();
        let optimizer = AdamW::new(3e-3).with_weight_decay(0.0).unwrap();
        let mut trainer =
            Inkling2Trainer::new(&session, configuration, decoder, optimizer, expert_precision, 3).unwrap();
        let reports = (0..steps).map(|_| trainer.step(&batch).unwrap()).collect::<Vec<_>>();
        assert_eq!(trainer.step_count(), steps);
        assert!(trainer.decoder().is_some());
        reports
    }

    #[test]
    fn test_inkling2_batch_new() {
        let batch = Inkling2Batch::new(Inkling2Inputs::new(1, 2, 3, 4), 5, 6);
        assert_eq!(batch.inputs(), &Inkling2Inputs::new(1, 2, 3, 4));
        assert_eq!(batch.labels(), &5);
        assert_eq!(batch.weights(), &6);
    }

    #[test]
    fn test_inkling2_batch_synthetic() {
        let configuration = test_configuration();
        let batch = Inkling2Batch::synthetic(&configuration, 2, 3, 7).unwrap();
        let tokens = batch.inputs().tokens().elements::<i32>().unwrap();
        assert_eq!(tokens.len(), 6);
        assert!(tokens.iter().all(|&token| (0..60).contains(&token)));
        assert_eq!(batch.inputs().positions().elements::<i32>().unwrap(), vec![0, 1, 2, 0, 1, 2]);
        assert_eq!(batch.inputs().segments().elements::<i32>().unwrap(), vec![0; 6]);
        assert_eq!(batch.inputs().valid().elements::<bool>().unwrap(), vec![true; 6]);

        // Every position is supervised with the token that follows it, except the last position of a sequence.
        assert_eq!(batch.labels().elements::<i32>().unwrap(), vec![tokens[1], tokens[2], 0, tokens[4], tokens[5], 0]);
        assert_eq!(batch.weights().elements::<f32>().unwrap(), vec![1.0, 1.0, 0.0, 1.0, 1.0, 0.0]);
        assert_eq!(batch, Inkling2Batch::synthetic(&configuration, 2, 3, 7).unwrap());

        assert!(matches!(
            Inkling2Batch::synthetic(&configuration, 1, 1, 0),
            Err(Inkling2TrainingError::InvalidBatch { message })
                if message == "a synthetic batch needs at least one sequence of at least two tokens but got 1 \
                               sequences of 1 tokens",
        ));
    }

    #[test]
    fn test_inkling2_trainer_step() {
        // The first step compiles the training step and the following ones reuse it, and training on one batch
        // lowers its loss.
        let reports = train(Inkling2ExpertPrecision::Floating, 8);
        assert_eq!(reports.iter().map(Inkling2StepReport::compiled).collect::<Vec<_>>()[..2], [true, false]);
        assert!(reports.iter().all(|report| report.loss().is_finite() && report.seconds() > 0.0));
        assert!(reports[7].loss() < reports[0].loss(), "{} is not below {}", reports[7].loss(), reports[0].loss());
    }

    #[cfg(feature = "safetensors")]
    #[test]
    fn test_inkling2_trainer_save() {
        // Restoring a checkpoint brings back the parameters, the optimizer state, and the step count, so the steps
        // after loading a checkpoint repeat the steps that followed the checkpoint when it was saved.
        let configuration = test_configuration();
        let session = Session::builder().platform(Platform::Cpu).build().unwrap();
        let decoder = Inkling2Decoder::random(&configuration, 1).unwrap();
        let batch = Inkling2Batch::synthetic(&configuration, 1, 32, 2).unwrap();
        let optimizer = AdamW::new(3e-3).with_weight_decay(0.0).unwrap();
        let mut trainer =
            Inkling2Trainer::new(&session, configuration, decoder, optimizer, Inkling2ExpertPrecision::Floating, 3)
                .unwrap();
        let directory = tempfile::tempdir().unwrap();
        let options = CheckpointerOptions::new()
            .with_save_decision_policy(FixedIntervalPolicy::new(2).unwrap())
            .with_preservation_policy(LatestN::new(1));
        let context = trainer.checkpoint_context(Arc::new(SafetensorsFormat));
        let mut checkpointer = Checkpointer::new(directory.path(), context, options).unwrap();

        // Steps that are not on the interval are not saved, and only the latest checkpoint is kept.
        let mut losses = Vec::new();
        for _ in 0..4 {
            let report = trainer.step(&batch).unwrap();
            losses.push(report.loss());
            let saved = trainer.save(&mut checkpointer, Some(report.loss())).unwrap();
            assert_eq!(saved, trainer.step_count().is_multiple_of(2));
        }
        assert_eq!(checkpointer.checkpoints().iter().map(|checkpoint| checkpoint.step()).collect::<Vec<_>>(), vec![4]);
        assert_eq!(checkpointer.latest().unwrap().metric("loss"), Some(f64::from(losses[3])));
        let continued = (0..2).map(|_| trainer.step(&batch).unwrap().loss()).collect::<Vec<_>>();
        assert_eq!(trainer.step_count(), 6);

        assert_eq!(trainer.load(&mut checkpointer, None).unwrap(), 4);
        assert_eq!(trainer.step_count(), 4);
        let repeated = (0..2).map(|_| trainer.step(&batch).unwrap().loss()).collect::<Vec<_>>();
        assert_eq!(repeated, continued);

        // A step without a checkpoint cannot be loaded.
        assert!(matches!(
            trainer.load(&mut checkpointer, Some(2)),
            Err(Inkling2TrainingError::Checkpoint(CheckpointError::MissingCheckpoint { .. })),
        ));
    }

    #[test]
    fn test_inkling2_trainer_step_block_scaled() {
        // The mixed precision recipe trains as well, and it starts from a different loss because its forward pass
        // quantizes the expert projections.
        let reports = train(Inkling2ExpertPrecision::BlockScaled, 8);
        assert!(reports.iter().all(|report| report.loss().is_finite()));
        assert!(reports[7].loss() < reports[0].loss(), "{} is not below {}", reports[7].loss(), reports[0].loss());
        assert_ne!(reports[0].loss(), train(Inkling2ExpertPrecision::Floating, 1)[0].loss());
    }
}
