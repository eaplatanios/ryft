# **Ryft Checkpoints:** Modular and Customizable Checkpointing Support for Ryft

`ryft-checkpoints` saves, loads, and imports the parameter structures of [`ryft-core`](../ryft-core). It follows the v1 API
of [Orbax](https://orbax.readthedocs.io/): free functions (`save`, `save_async`, `load`, and their checkpointables
counterparts) that take an explicit `Context`, checkpoints of named checkpointables that handlers save and load, and a
`training::Checkpointer` that manages the atomically saved step checkpoints of a training run (i.e., save decision and
preservation policies and asynchronous and multi-process saving). It reads and writes the safetensors file format
(behind the default `safetensors` feature), pins released checkpoints with manifests, decodes packed low precision
weights, and defines how model code maps a foreign checkpoint to its parameters.

By default, this crate depends on `ryft-core` only among the Ryft crates. It works with host arrays and has no XLA,
PJRT, or accelerator dependency. Checkpoints live on the local file system or, through the `FileSystem` abstraction,
in other storage. These features change what it includes:

- `safetensors` (enabled by default) stores the arrays of checkpoints in the safetensors file format. Without it, the
  storage format is whatever implementation of `ArrayFormat` the caller provides.
- `xla` adds a dependency on [`ryft-xla`](../ryft-xla), a leaf handler that loads checkpoints directly into the
  buffers of XLA devices, with a sharding per leaf, and reads them back for saving, and a process barrier over the
  coordination service of a distributed XLA runtime. The platform features of `ryft-xla` (i.e., `cuda-12`, `cuda-13`,
  `rocm-10`, `tpu`, `neuron`, `metal`, and `mps`) are forwarded under the same names and enable `xla`.
- `object-store` (disabled by default) adds a dependency on [`object_store`](https://docs.rs/object_store) and a file
  system that stores checkpoints in S3, GCS, Azure Blob Storage, or any other object store, finalizing them with commit
  files because object stores cannot rename atomically. Cloud backends are enabled through the `aws`, `gcp`, and `azure`
  features of `object_store` in the depending crate.
- `zarr` (disabled by default) stores the arrays of checkpoints as uncompressed [Zarr](https://zarr.dev/) version 3
  arrays, one group per checkpointable and one array per leaf, which is the layout that TensorStore writes for Orbax
  without [OCDBT](https://google.github.io/tensorstore/kvstore/ocdbt/index.html).

## Example

The following imports the arrays of a Hugging Face safetensors checkpoint, keeps the latest two step checkpoints of
a training run that starts from them, and loads the latest one back. It works with the `model.safetensors` file of
any model on the Hugging Face Hub, and it is the body of the [`hugging_face`](examples/hugging_face.rs) example:

```sh
hf download openai-community/gpt2 model.safetensors --local-dir gpt2
cargo run --release -p ryft-checkpoints --example hugging_face -- gpt2/model.safetensors runs/gpt2
```

```rust
use std::collections::BTreeMap;

use ryft_checkpoints::training::{Checkpointer, CheckpointerOptions, LatestN};
use ryft_checkpoints::{
    ArraySpecification, Checkpoint, CheckpointError, Context, SafetensorsCheckpoint, SaveOptions, import_arrays,
};
use ryft_core::{Array, ArrayType};

let (model_path, run_directory) = ("gpt2/model.safetensors", "runs/gpt2");

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
```

Every checkpoint is a directory of named checkpointables that is finalized atomically, so the run directory above
holds `<step>/_CHECKPOINT_METADATA` and `<step>/state/` for each kept step. A real model describes its parameters as
a struct that derives `Parameterized` instead of a map, and maps the array names of a released checkpoint to its
leaves in an `Importer`. Checkpoints that are not tied to training steps are saved and loaded with the `save` and
`load` functions, and `save_checkpointables` stores further named values (e.g., a JSON configuration) next to the
parameters.

#### License

<sup>
Licensed under either <a href="../../LICENSE-APACHE">Apache License, Version 2.0</a>
or <a href="../../LICENSE-MIT">MIT license</a> at your option.
</sup>

<br>

<sub>
Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this crate by you,
as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
</sub>
