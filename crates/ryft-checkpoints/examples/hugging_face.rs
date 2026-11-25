//! Imports the arrays of a Hugging Face safetensors checkpoint and keeps step checkpoints of a training run that
//! starts from them.
//!
//! The example works with the `model.safetensors` file of any model on the Hugging Face Hub (e.g., GPT-2, which
//! `hf download openai-community/gpt2 model.safetensors --local-dir gpt2` downloads):
//!
//! ```sh
//! cargo run --release -p ryft-checkpoints --example hugging_face -- gpt2/model.safetensors runs/gpt2
//! ```
//!
//! It imports every array of the file into a map from array names to host arrays. A real model instead describes
//! its parameters as a struct that derives `Parameterized` and maps the names of the file to its leaves in an
//! `Importer`, which is also where packed or renamed arrays are decoded.

use std::collections::BTreeMap;
use std::error::Error;

use ryft_checkpoints::training::{Checkpointer, CheckpointerOptions, LatestN};
use ryft_checkpoints::{
    ArraySpecification, Checkpoint, CheckpointError, Context, SafetensorsCheckpoint, SaveOptions, import_arrays,
};
use ryft_core::{Array, ArrayType};

// TODO(eaplatanios): Review this module.

fn main() -> Result<(), Box<dyn Error>> {
    let arguments = std::env::args().collect::<Vec<_>>();
    let [_, model_path, run_directory] = arguments.as_slice() else {
        return Err("usage: hugging_face <model.safetensors> <run directory>".into());
    };
    let (model_path, run_directory) = (model_path.as_str(), run_directory.as_str());

    // Opening the file reads only its header, which names every array and gives its data type and shape.
    let checkpoint = SafetensorsCheckpoint::open(model_path)?;
    for array in checkpoint.arrays() {
        println!("{:<48} {:<5} {:?}", array.name(), array.data_type().to_string(), array.shape());
    }

    // The leaves to import are described by a parameter structure of array specifications. `import_arrays` reads
    // one array at a time, checks its type, and hands it to the last function, which decides where the leaf lives
    // (e.g., it could upload the array to an accelerator and drop the host copy).
    let specifications = checkpoint
        .arrays()
        .iter()
        .map(|array| (array.name().to_string(), ArraySpecification::new(array.name(), array.array_type())))
        .collect::<BTreeMap<_, _>>();
    let state: BTreeMap<String, Array> = import_arrays(
        specifications,
        |specification| checkpoint.read(specification.name()),
        |_path, array| Ok::<_, CheckpointError>(array),
    )?;

    // A training checkpointer keeps one checkpoint per saved step in the run directory. Its policies decide which
    // steps are saved and which checkpoints are kept, and every checkpoint is finalized atomically.
    let options = CheckpointerOptions::new().with_preservation_policy(LatestN::new(2));
    let mut checkpointer = Checkpointer::new(run_directory, Context::default(), options)?;
    let first_step = checkpointer.latest().map_or(1, |checkpoint| checkpoint.step() + 1);
    for step in first_step..first_step + 3 {
        // A training step would update `state` here.
        let metrics = BTreeMap::from([("loss".to_string(), 1.0 / step as f64)]);
        if checkpointer.save(step, &state, &SaveOptions::new().with_metrics(metrics))? {
            println!("saved the checkpoint of step {step}");
        }
    }

    // Loading is driven by an abstract state, which has the structure of the state and an array type in place of
    // every leaf. With `None` for the step, the latest checkpoint is loaded.
    let abstract_state = checkpoint
        .arrays()
        .iter()
        .map(|array| (array.name().to_string(), array.array_type()))
        .collect::<BTreeMap<String, ArrayType>>();
    let loaded = checkpointer.load(None, abstract_state)?;
    println!("loaded {} arrays from the latest checkpoint", loaded.len());
    for checkpoint in checkpointer.checkpoints() {
        println!("step {} at `{}` with metrics {:?}", checkpoint.step(), checkpoint.path(), checkpoint.metrics());
    }
    Ok(())
}
