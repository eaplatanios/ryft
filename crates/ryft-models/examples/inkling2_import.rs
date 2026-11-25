//! Checks a downloaded Inkling 2 Mini release against the pinned manifest and imports its decoder parameters.
//!
//! The example takes the directory that holds the files of the release (at least `config.json` and
//! `model-00001-of-00001.safetensors`) and performs the following checks in order, stopping at the first one that
//! fails.
//!
//!   1. The checkpoint file has the size and, unless `--skip-digest` is passed, the SHA-256 digest that the pinned
//!      [`Manifest`] records.
//!   2. The checkpoint holds as many arrays as the release.
//!   3. The `config.json` of the release parses to the Inkling 2 Mini configuration.
//!   4. The checkpoint holds every array of the decoder in a supported storage format and no unaccounted one.
//!   5. Every leaf of the decoder imports. Leaves are streamed one at a time and dropped after their statistics are
//!      taken, so the import needs the memory of the largest leaf rather than of the whole decoder.
//!   6. For the first and the last expert of every packed projection (or for as many leading and trailing experts as
//!      `--round-trip-experts` selects), encoding the imported values again reproduces the stored bytes.
//!
//! It prints one JSON object with what it found.
//!
//! ```sh
//! cargo run --release -p ryft-models --example inkling2_import -- <release directory>
//! ```

use std::collections::HashMap;
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

use ryft_checkpoints::codecs::{Nvfp4OuterScale, Nvfp4ScaleSelection, encode_mxfp8, encode_nvfp4};
use ryft_checkpoints::{Checkpoint, CheckpointError, FileRecord, Importer, SafetensorsCheckpoint};
use ryft_core::{Array, DataType, Parameterized, Typed};
use ryft_models::inkling2::{
    INKLING_2_MINI_ARRAY_COUNT, INKLING_2_MINI_CHECKPOINT_FILE, Inkling2Configuration, Inkling2Decoder,
    Inkling2Importer, inkling_2_mini_manifest,
};

// TODO(eaplatanios): Review this module.

/// Statistics of the imported leaves.
#[derive(Default)]
struct Statistics {
    /// Number of imported leaves.
    leaves: usize,

    /// Number of imported scalar parameters.
    parameters: u64,

    /// Number of imported scalar parameters of expert projections.
    expert_parameters: u64,

    /// Number of imported values that are not finite.
    non_finite: u64,

    /// Largest magnitude of the imported values.
    largest_magnitude: f32,

    /// Number of experts whose imported values were encoded again and compared with the stored bytes.
    round_trip_experts: usize,

    /// Names of the arrays for which an encoded expert differed from the stored bytes.
    round_trip_mismatches: Vec<String>,
}

/// Encodes the imported values of expert `expert` of the packed projection `name` again and returns whether the
/// result equals the stored element and scale bytes.
fn round_trips<C: Checkpoint>(
    checkpoint: &C,
    name: &str,
    expert: usize,
    values: &[f32],
) -> Result<bool, Box<dyn Error>> {
    let stored_elements = checkpoint.read_leading(name, expert..expert + 1)?;
    let stored_scales = checkpoint.read_leading(&format!("{name}_scale"), expert..expert + 1)?;
    let (elements, scales) = match stored_elements.r#type().data_type() {
        DataType::U8 => {
            let amax = checkpoint.read(&format!("{name}_amax"))?.elements::<f32>()?[expert];
            let denominator = checkpoint.read(&format!("{name}_outer_scale_denominator"))?.elements::<f32>()?[0];
            let outer_scale = Nvfp4OuterScale::for_weights(amax, denominator)?;
            encode_nvfp4(values, outer_scale, Nvfp4ScaleSelection::Six)?
        }
        _ => encode_mxfp8(values)?,
    };
    Ok(elements == stored_elements.storage_bytes() && scales == stored_scales.storage_bytes())
}

/// Runs the checks that the command line arguments describe.
fn main() -> Result<(), Box<dyn Error>> {
    let (mut directory, mut skip_digest, mut round_trip_experts) = (None, false, 1usize);
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--skip-digest" => skip_digest = true,
            "--round-trip-experts" => {
                round_trip_experts = arguments.next().ok_or("option `--round-trip-experts` needs a value")?.parse()?;
            }
            _ if directory.is_none() => directory = Some(PathBuf::from(argument)),
            _ => return Err(format!("unexpected argument `{argument}`").into()),
        }
    }
    let directory = directory.ok_or("usage: inkling2_import <release directory> [--skip-digest]")?;
    let checkpoint_path = directory.join(INKLING_2_MINI_CHECKPOINT_FILE);
    let manifest = inkling_2_mini_manifest();

    // The digest check reads the whole file once, so it is the only step that can be skipped.
    let start = Instant::now();
    let manifest = if skip_digest {
        let files = manifest.files().iter().map(|file| FileRecord::new(file.path(), file.byte_count()));
        manifest.clone().with_files(files.collect())
    } else {
        manifest
    };
    manifest.validate_file(INKLING_2_MINI_CHECKPOINT_FILE, &checkpoint_path)?;
    let digest_seconds = start.elapsed().as_secs_f64();
    eprintln!("checkpoint file matches the manifest ({digest_seconds:.1} s)");

    let checkpoint = SafetensorsCheckpoint::open(&checkpoint_path)?;
    if checkpoint.arrays().len() != INKLING_2_MINI_ARRAY_COUNT {
        return Err(format!(
            "the checkpoint holds {} arrays but the release holds {INKLING_2_MINI_ARRAY_COUNT}",
            checkpoint.arrays().len(),
        )
        .into());
    }
    eprintln!("checkpoint holds the {} arrays of the release", checkpoint.arrays().len());

    let configuration = Inkling2Configuration::inkling_2_mini();
    let released = Inkling2Configuration::from_json(&std::fs::read_to_string(directory.join("config.json"))?)?;
    if released.with_nvfp4_layer_count(configuration.nvfp4_layer_count()) != configuration {
        return Err("`config.json` does not describe the Inkling 2 Mini configuration".into());
    }
    let importer = Inkling2Importer::new(configuration.clone());
    importer.validate(&checkpoint)?;
    eprintln!("checkpoint holds every decoder array and no unaccounted one");

    let specification = Inkling2Decoder::specification(&configuration);
    let names = specification
        .named_parameters()
        .map(|(path, leaf)| (path.to_string(), leaf.name().to_string()))
        .collect::<HashMap<_, _>>();
    let mut statistics = Statistics::default();
    let start = Instant::now();
    let norms = importer.import(&checkpoint, |path, array: Array| {
        let name = &names[&path.to_string()];
        let shape = array.r#type().static_shape().map(|shape| shape.dimensions().to_vec()).unwrap_or_default();
        let values = array
            .storage_bytes()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| f32::from_ne_bytes(*bytes))
            .collect::<Vec<_>>();
        let mut squares = 0.0f64;
        for &value in &values {
            if !value.is_finite() {
                statistics.non_finite += 1;
            }
            statistics.largest_magnitude = statistics.largest_magnitude.max(value.abs());
            squares += f64::from(value) * f64::from(value);
        }
        statistics.leaves += 1;
        statistics.parameters += values.len() as u64;

        // Packed projections are the leaves whose checkpoint array has block scales.
        if checkpoint.array(&format!("{name}_scale")).is_some() {
            statistics.expert_parameters += values.len() as u64;
            let expert_size = shape[1] * shape[2];
            let selected =
                (0..shape[0]).filter(|&expert| expert < round_trip_experts || expert + round_trip_experts >= shape[0]);
            for expert in selected {
                let expert_values = &values[expert * expert_size..(expert + 1) * expert_size];
                let matches = round_trips(&checkpoint, name, expert, expert_values)
                    .map_err(|error| CheckpointError::Import { message: error.to_string() })?;
                statistics.round_trip_experts += 1;
                if !matches {
                    statistics.round_trip_mismatches.push(format!("{name}[{expert}]"));
                }
            }
        }
        if statistics.leaves % 50 == 0 {
            eprintln!("imported {} leaves ({:.0} s)", statistics.leaves, start.elapsed().as_secs_f64());
        }
        Ok(squares.sqrt())
    })?;
    let import_seconds = start.elapsed().as_secs_f64();

    println!(
        "{}",
        serde_json::json!({
            "repository": manifest.repository(),
            "revision": manifest.revision(),
            "digest_checked": !skip_digest,
            "digest_seconds": digest_seconds,
            "arrays": checkpoint.arrays().len(),
            "leaves": statistics.leaves,
            "parameters": statistics.parameters,
            "expert_parameters": statistics.expert_parameters,
            "non_finite_values": statistics.non_finite,
            "largest_magnitude": statistics.largest_magnitude,
            "parameter_norm": norms.parameters().map(|norm| norm * norm).sum::<f64>().sqrt(),
            "round_trip_experts": statistics.round_trip_experts,
            "round_trip_mismatches": statistics.round_trip_mismatches,
            "import_seconds": import_seconds,
        }),
    );
    if statistics.non_finite > 0 || !statistics.round_trip_mismatches.is_empty() {
        return Err("the imported parameters failed a check".into());
    }
    Ok(())
}
