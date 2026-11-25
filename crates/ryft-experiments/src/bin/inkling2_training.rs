//! Times supervised training steps of the Inkling 2 text decoder on an XLA platform.
//!
//! The binary builds a decoder with reproducible pseudo-random parameters, trains it on one synthetic batch with
//! AdamW, and prints one JSON object with the configuration of the run, the duration of the first step (which
//! compiles the training step), the durations of the following steps, and the loss of every step.
//!
//! ```sh
//! cargo run --release -p ryft-experiments --bin inkling2_training -- --platform cpu
//! cargo run --release -p ryft-experiments --features mps --bin inkling2_training -- --platform mps
//! cargo run --release -p ryft-experiments --features cuda-13 --bin inkling2_training -- --platform cuda
//! ```
//!
//! With `--checkpoint-directory`, the run saves its training state (i.e., the parameters and the optimizer state)
//! through a [`Checkpointer`] every `--checkpoint-interval` steps, keeps the latest `--keep-checkpoints` checkpoints,
//! and continues from the latest checkpoint of the directory when `--resume` is passed. The arrays are stored in the
//! format that `--checkpoint-format` selects at runtime, which needs the feature of the same name (`safetensors`,
//! which is enabled by default, or `zarr`).

use std::error::Error;
use std::sync::Arc;

use ryft_checkpoints::ArrayFormat;
#[cfg(feature = "safetensors")]
use ryft_checkpoints::SafetensorsFormat;
#[cfg(feature = "zarr")]
use ryft_checkpoints::ZarrFormat;
use ryft_checkpoints::training::{Checkpointer, CheckpointerOptions, FixedIntervalPolicy, LatestN};
use ryft_core::DataType;
use ryft_experiments::inkling2::{Inkling2Batch, Inkling2ExpertPrecision, Inkling2Trainer};
use ryft_models::inkling2::{Inkling2Configuration, Inkling2Decoder};
use ryft_optimizers::AdamW;
use ryft_xla::{Platform, Session};

// TODO(eaplatanios): Review this module.

/// Usage text of this binary.
const USAGE: &str = "\
usage: inkling2_training [options]
  --platform <cpu|mps|metal|cuda|auto>   XLA platform that executes the training step (default: auto)
  --compute-type <bf16|f32>              compute data type of the decoder (default: bf16)
  --expert-precision <floating|block-scaled>   precision of the expert projections (default: floating)
  --batch-size <count>                   number of sequences per batch (default: 2)
  --sequence-length <count>              number of tokens per sequence (default: 64)
  --steps <count>                        number of training steps, including the compiling one (default: 6)
  --learning-rate <value>                AdamW learning rate (default: 1e-3)
  --seed <value>                         seed of the parameters and of the batch (default: 0)
  --checkpoint-directory <path>          directory in which the training state is checkpointed (default: none)
  --checkpoint-format <safetensors|zarr> storage format of the arrays of checkpoints (default: safetensors)
  --checkpoint-interval <count>          number of steps between checkpoints (default: 1)
  --keep-checkpoints <count>             number of latest checkpoints to keep (default: all)
  --resume                               continue from the latest checkpoint of the directory";

/// Parses the value of the command line option `name`.
fn parse<T: std::str::FromStr>(name: &str, value: Option<String>) -> Result<T, Box<dyn Error>> {
    let value = value.ok_or_else(|| format!("option `{name}` needs a value"))?;
    value.parse().map_err(|_| format!("invalid value `{value}` for option `{name}`").into())
}

/// Runs the experiment that the command line arguments describe.
fn main() -> Result<(), Box<dyn Error>> {
    let mut platform = Platform::Auto;
    let mut compute_type = DataType::BF16;
    let mut expert_precision = Inkling2ExpertPrecision::Floating;
    let (mut batch_size, mut sequence_length, mut steps, mut learning_rate, mut seed) =
        (2usize, 64usize, 6usize, 1e-3, 0);
    let (mut checkpoint_directory, mut checkpoint_interval, mut keep_checkpoints, mut resume) =
        (None::<String>, 1u64, None::<usize>, false);
    let mut checkpoint_format = "safetensors".to_string();
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--platform" => {
                platform = match parse::<String>(&argument, arguments.next())?.as_str() {
                    "auto" => Platform::Auto,
                    "cpu" => Platform::Cpu,
                    "mps" => Platform::Mps,
                    "metal" => Platform::Metal,
                    "cuda" => Platform::Cuda,
                    platform => return Err(format!("unknown platform `{platform}`").into()),
                };
            }
            "--compute-type" => {
                compute_type = match parse::<String>(&argument, arguments.next())?.as_str() {
                    "bf16" => DataType::BF16,
                    "f32" => DataType::F32,
                    compute_type => return Err(format!("unknown compute data type `{compute_type}`").into()),
                };
            }
            "--expert-precision" => {
                expert_precision = match parse::<String>(&argument, arguments.next())?.as_str() {
                    "floating" => Inkling2ExpertPrecision::Floating,
                    "block-scaled" => Inkling2ExpertPrecision::BlockScaled,
                    precision => return Err(format!("unknown expert precision `{precision}`").into()),
                };
            }
            "--batch-size" => batch_size = parse(&argument, arguments.next())?,
            "--sequence-length" => sequence_length = parse(&argument, arguments.next())?,
            "--steps" => steps = parse(&argument, arguments.next())?,
            "--learning-rate" => learning_rate = parse(&argument, arguments.next())?,
            "--seed" => seed = parse(&argument, arguments.next())?,
            "--checkpoint-directory" => checkpoint_directory = Some(parse(&argument, arguments.next())?),
            "--checkpoint-format" => checkpoint_format = parse(&argument, arguments.next())?,
            "--checkpoint-interval" => checkpoint_interval = parse(&argument, arguments.next())?,
            "--keep-checkpoints" => keep_checkpoints = Some(parse(&argument, arguments.next())?),
            "--resume" => resume = true,
            "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            argument => return Err(format!("unknown argument `{argument}`\n{USAGE}").into()),
        }
    }

    let configuration = Inkling2Configuration::reduced().with_compute_type(compute_type);
    let session = Session::builder().platform(platform).build()?;
    let platform_name = session.target()?.platform_name().to_string();
    eprintln!("platform: {platform_name}");
    let decoder = Inkling2Decoder::random(&configuration, seed)?;
    let parameter_count = decoder_parameter_count(&decoder);
    let batch = Inkling2Batch::synthetic(&configuration, batch_size, sequence_length, seed)?;
    let optimizer = AdamW::new(learning_rate).with_weight_decay(0.0)?;
    let mut trainer = Inkling2Trainer::new(&session, configuration, decoder, optimizer, expert_precision, seed)?;

    let mut checkpointer = match &checkpoint_directory {
        Some(directory) => {
            let mut options =
                CheckpointerOptions::new().with_save_decision_policy(FixedIntervalPolicy::new(checkpoint_interval)?);
            if let Some(count) = keep_checkpoints {
                options = options.with_preservation_policy(LatestN::new(count));
            }
            let context = trainer.checkpoint_context(array_format(&checkpoint_format)?);
            let mut checkpointer = Checkpointer::new(directory.as_str(), context, options)?;
            if resume && checkpointer.latest().is_some() {
                let step = trainer.load(&mut checkpointer, None)?;
                eprintln!("resumed from the checkpoint of step {step}");
            }
            Some(checkpointer)
        }
        None if resume => return Err("option `--resume` needs `--checkpoint-directory`".into()),
        None => None,
    };

    let mut losses = Vec::with_capacity(steps);
    let mut seconds = Vec::with_capacity(steps);
    for _ in 0..steps {
        let report = trainer.step(&batch)?;
        let step = trainer.step_count();
        eprintln!("step {step}: loss = {:.6}, {:.3} s", report.loss(), report.seconds());
        losses.push(report.loss());
        seconds.push(report.seconds());
        if let Some(checkpointer) = &mut checkpointer
            && trainer.save(checkpointer, Some(report.loss()))?
        {
            eprintln!("saved the checkpoint of step {step}");
        }
    }

    let warm = &seconds[1.min(seconds.len())..];
    let mut sorted = warm.to_vec();
    sorted.sort_by(f64::total_cmp);
    let median = sorted.get(sorted.len() / 2).copied();
    println!(
        "{}",
        serde_json::json!({
            "platform": platform_name,
            "compute_type": compute_type.to_string(),
            "expert_precision": format!("{expert_precision:?}"),
            "parameter_count": parameter_count,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "first_step_seconds": seconds.first(),
            "warm_step_seconds": warm,
            "median_warm_step_seconds": median,
            "losses": losses,
        }),
    );
    Ok(())
}

/// Returns the error for a checkpoint format whose feature is not enabled.
#[cfg(not(all(feature = "safetensors", feature = "zarr")))]
fn missing_feature(format: &str) -> Box<dyn Error> {
    format!("checkpoint format `{format}` needs the `{format}` feature of `ryft-experiments`").into()
}

/// Returns the array format of checkpoints named `name`.
fn array_format(name: &str) -> Result<Arc<dyn ArrayFormat>, Box<dyn Error>> {
    match name {
        #[cfg(feature = "safetensors")]
        "safetensors" => Ok(Arc::new(SafetensorsFormat)),
        #[cfg(feature = "zarr")]
        "zarr" => Ok(Arc::new(ZarrFormat::new())),
        #[cfg(not(feature = "safetensors"))]
        "safetensors" => Err(missing_feature("safetensors")),
        #[cfg(not(feature = "zarr"))]
        "zarr" => Err(missing_feature("zarr")),
        name => Err(format!("unknown checkpoint format `{name}`").into()),
    }
}

/// Returns the number of scalar parameters of `decoder`.
fn decoder_parameter_count(decoder: &Inkling2Decoder<ryft_core::Array>) -> usize {
    use ryft_core::{Parameterized, Typed};
    decoder
        .parameters()
        .map(|leaf| leaf.r#type().static_shape().map_or(0, |shape| shape.dimensions().iter().product::<usize>()))
        .sum()
}
