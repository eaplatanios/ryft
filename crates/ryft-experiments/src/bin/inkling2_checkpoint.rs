//! Writes a synthetic Inkling 2 decoder checkpoint in the array layout of Inkling 2 releases.
//!
//! The checkpoint holds the reproducible pseudo-random parameters of [`Inkling2Decoder::random`] as `f32` arrays
//! under the array names of a release, so it can be read by the
//! [`Inkling2Importer`](ryft_models::inkling2::Inkling2Importer) and by reference implementations that read release
//! checkpoints. This is how the parity fixtures of the decoder are produced without the released weights.
//!
//! ```sh
//! cargo run -p ryft-experiments --bin inkling2_checkpoint -- --configuration config.json --output model.safetensors
//! ```

use std::collections::BTreeMap;
use std::error::Error;
use std::fs::File;
use std::io::BufWriter;

use ryft_checkpoints::write_safetensors;
use ryft_models::inkling2::{Inkling2Configuration, Inkling2Decoder, export_inkling2_decoder};

// TODO(eaplatanios): Review this module.

/// Usage text of this binary.
const USAGE: &str = "\
usage: inkling2_checkpoint --output <path> [options]
  --configuration <path>   `config.json` of the decoder (default: the reduced configuration)
  --nvfp4-layers <count>   number of leading NVFP4 layers of a configuration that is read from a file
  --seed <value>           seed of the parameters (default: 0)
  --output <path>          path of the safetensors file to write";

/// Writes the checkpoint that the command line arguments describe.
fn main() -> Result<(), Box<dyn Error>> {
    let (mut configuration, mut nvfp4_layers, mut seed, mut output) = (None, None, 0u64, None);
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        let mut value = || arguments.next().ok_or_else(|| format!("option `{argument}` needs a value"));
        match argument.as_str() {
            "--configuration" => configuration = Some(value()?),
            "--nvfp4-layers" => nvfp4_layers = Some(value()?.parse::<usize>()?),
            "--seed" => seed = value()?.parse()?,
            "--output" => output = Some(value()?),
            "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            _ => return Err(format!("unknown argument `{argument}`\n{USAGE}").into()),
        }
    }
    let output = output.ok_or_else(|| format!("missing option `--output`\n{USAGE}"))?;
    let configuration = match configuration {
        None => Inkling2Configuration::reduced(),
        Some(path) => {
            let configuration = Inkling2Configuration::from_json(&std::fs::read_to_string(path)?)?;
            match nvfp4_layers {
                Some(nvfp4_layers) => configuration.with_nvfp4_layer_count(nvfp4_layers),
                None => configuration,
            }
        }
    };
    configuration.validate()?;
    let decoder = Inkling2Decoder::random(&configuration, seed)?;
    let arrays = export_inkling2_decoder(&configuration, &decoder)?;
    write_safetensors(BufWriter::new(File::create(&output)?), &arrays, &BTreeMap::new())?;
    eprintln!("wrote {} arrays to {output}", arrays.len());
    Ok(())
}
