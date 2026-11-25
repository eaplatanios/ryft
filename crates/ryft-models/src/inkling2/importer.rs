//! Import and export of Inkling 2 decoder parameters in the array layout of Inkling 2 release checkpoints.
//!
//! A release checkpoint stores most arrays as `bf16` or `f32` arrays and the two projections of the experts of every
//! layer in a packed block-scaled format, which is NVFP4 for the leading layers and MXFP8 for the remaining ones
//! (refer to the [`codecs`](ryft_checkpoints::codecs) module for both formats). A packed projection named `w` comes
//! with the block scales `w_scale` and, for NVFP4, with the per expert largest magnitudes `w_amax` and the scalar
//! `w_outer_scale_denominator`, whose quotient is the outer scale of an expert. Packed projections have the layout
//! `[experts, outputs, inputs]` with blocks along the input axis.
//!
//! [`Inkling2Importer`] maps such a checkpoint to an [`Inkling2Decoder`] whose leaves are `f32` master parameters.
//! Floating-point arrays are converted to `f32`, which is exact, and packed projections are decoded to the `f32`
//! values that they represent, one expert at a time so that only one expert is expanded outside of the leaf that is
//! being assembled. The storage format of a projection is recognized from the data type of its array rather than
//! from the configuration, so the same importer reads the floating-point checkpoints that [`export_inkling2_decoder`]
//! writes. Before anything is read, the importer checks that the checkpoint holds every array that the decoder
//! needs and no language model array that it does not account for. Arrays of the vision encoder (i.e., those whose
//! names start with `model.visual.`) are recognized and left unread.

use std::collections::HashSet;

use ryft_checkpoints::codecs::{
    CodecError, MXFP8_BLOCK_SIZE, NVFP4_BLOCK_SIZE, Nvfp4OuterScale, decode_mxfp8, decode_nvfp4,
};
use ryft_checkpoints::{
    ArraySpecification, Checkpoint, CheckpointError, FileRecord, Importer, Manifest, import_arrays,
};
use ryft_core::{Array, ArrayType, DataType, Parameter, ParameterPath, Parameterized, Typed};

use crate::inkling2::configuration::Inkling2Configuration;
use crate::inkling2::decoder::Inkling2Decoder;

// TODO(eaplatanios): Review this module.

/// Prefix of the names of the arrays of the vision encoder, which the decoder importer leaves unread.
const VISION_ARRAY_PREFIX: &str = "model.visual.";

/// Name of the checkpoint file of the Inkling 2 Mini release.
pub const INKLING_2_MINI_CHECKPOINT_FILE: &str = "model-00001-of-00001.safetensors";

/// Number of arrays in the checkpoint file of the Inkling 2 Mini release, which are the 694 arrays of the decoder,
/// the 128 sidecars of its packed projections, and the 422 arrays of the vision encoder.
pub const INKLING_2_MINI_ARRAY_COUNT: usize = 1244;

/// Returns the [`Manifest`] of the Inkling 2 Mini release that this module was written against, which pins the
/// repository, the revision, and the size and SHA-256 digest of the checkpoint file. It records no array table.
/// [`Inkling2Importer::validate`] checks the arrays of a checkpoint against the configuration instead, and
/// [`INKLING_2_MINI_ARRAY_COUNT`] is the number of arrays of the released file.
pub fn inkling_2_mini_manifest() -> Manifest {
    Manifest::new("thinkingmachines/Inkling-2-Mini", "ed40e7da625f594fc9c4990f5f66be24873c22d7").with_files(vec![
        FileRecord::new(INKLING_2_MINI_CHECKPOINT_FILE, 19_526_019_168)
            .with_sha256("65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1"),
    ])
}

/// [`Importer`] of the parameters of an [`Inkling2Decoder`] from a checkpoint in the array layout of Inkling 2
/// releases. Refer to the [module documentation](self) for the formats that it reads.
#[derive(Clone, Debug, PartialEq)]
pub struct Inkling2Importer {
    /// Configuration of the decoder whose parameters are imported.
    configuration: Inkling2Configuration,
}

impl Inkling2Importer {
    /// Creates a new [`Inkling2Importer`] for the decoder that `configuration` describes.
    #[inline]
    pub fn new(configuration: Inkling2Configuration) -> Self {
        Self { configuration }
    }

    /// Returns the configuration of the decoder whose parameters are imported.
    #[inline]
    pub fn configuration(&self) -> &Inkling2Configuration {
        &self.configuration
    }

    /// Checks that `checkpoint` holds the array of every leaf of the decoder with a supported storage format and
    /// the shape that the configuration requires, together with the scale arrays of every packed projection, and
    /// that it holds no other language model array.
    ///
    /// # Errors
    ///
    /// Returns [`CheckpointError::Import`] that names the first missing, malformed, or unaccounted array.
    pub fn validate<C: Checkpoint>(&self, checkpoint: &C) -> Result<(), CheckpointError> {
        let specification = Inkling2Decoder::specification(&self.configuration);
        let mut expected = HashSet::new();
        for leaf in specification.parameters() {
            for name in required_arrays(checkpoint, leaf)? {
                expected.insert(name);
            }
        }
        for array in checkpoint.arrays() {
            if !array.name().starts_with(VISION_ARRAY_PREFIX) && !expected.contains(array.name()) {
                return Err(CheckpointError::Import {
                    message: format!("checkpoint array `{}` is not an array of the Inkling 2 decoder", array.name()),
                });
            }
        }
        Ok(())
    }
}

impl Importer for Inkling2Importer {
    type Parameters<P: Parameter> = Inkling2Decoder<P>;

    fn import<C: Checkpoint, P: Parameter, F: FnMut(&ParameterPath, Array) -> Result<P, CheckpointError>>(
        &self,
        checkpoint: &C,
        place: F,
    ) -> Result<Self::Parameters<P>, CheckpointError> {
        self.validate(checkpoint)?;
        import_arrays(
            Inkling2Decoder::specification(&self.configuration),
            |specification| read_array(checkpoint, specification),
            place,
        )
    }
}

/// Returns the arrays of `decoder` under the names and in the layouts of an Inkling 2 release checkpoint, with every
/// array stored as `f32`. Writing them with `ryft_checkpoints::write_safetensors` produces a checkpoint that
/// [`Inkling2Importer`] reads back without loss.
///
/// # Errors
///
/// Returns [`CheckpointError::InvalidArray`] if a leaf of `decoder` does not have the type that `configuration`
/// requires, and [`CheckpointError::Parameter`] if `decoder` does not have the structure that it describes.
pub fn export_inkling2_decoder(
    configuration: &Inkling2Configuration,
    decoder: &Inkling2Decoder<Array>,
) -> Result<Vec<(String, Array)>, CheckpointError> {
    let specification = Inkling2Decoder::specification(configuration);
    let expected_paths = specification.parameter_paths().collect::<Vec<_>>();
    decoder.validate_parameter_structure(&expected_paths)?;
    specification
        .parameters()
        .zip(decoder.parameters())
        .map(|(specification, leaf)| {
            if leaf.r#type().as_ref() != specification.r#type() {
                return Err(CheckpointError::InvalidArray {
                    name: specification.name().to_string(),
                    message: format!(
                        "leaf must have type `{}` but has type `{}`",
                        specification.r#type(),
                        leaf.r#type(),
                    ),
                });
            }
            Ok((specification.name().to_string(), leaf.clone()))
        })
        .collect()
}

/// Storage format of the checkpoint array of a decoder leaf.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum StorageFormat {
    /// Floating-point array with the shape of the leaf.
    Floating,

    /// Packed NVFP4 projection with its block scales, per expert largest magnitudes, and outer scale denominator.
    Nvfp4,

    /// MXFP8 projection with its block scale exponents.
    Mxfp8,
}

/// Returns an import error for the array of `specification`.
fn import_error(specification: &ArraySpecification, message: String) -> CheckpointError {
    CheckpointError::Import { message: format!("checkpoint array `{}` {message}", specification.name()) }
}

/// Recognizes the storage format of the array of `specification` in `checkpoint` and checks the shapes and data
/// types of the array and of the scale arrays that its format requires.
fn storage_format<C: Checkpoint>(
    checkpoint: &C,
    specification: &ArraySpecification,
) -> Result<StorageFormat, CheckpointError> {
    let name = specification.name();
    let shape = specification.r#type().static_shape().map(|shape| shape.dimensions().to_vec()).unwrap_or_default();
    let array = checkpoint.array(name).ok_or_else(|| import_error(specification, "is missing".to_string()))?;
    let (format, divisor, block_size) = match array.data_type() {
        DataType::BF16 | DataType::F16 | DataType::F32 | DataType::F64 => (StorageFormat::Floating, 1, 1),
        DataType::U8 => (StorageFormat::Nvfp4, 2, NVFP4_BLOCK_SIZE),
        DataType::F8E4M3FN => (StorageFormat::Mxfp8, 1, MXFP8_BLOCK_SIZE),
        data_type => {
            return Err(import_error(specification, format!("has unsupported data type `{data_type}`")));
        }
    };

    // Packed formats store rank-3 projections with blocks along their trailing axis.
    let packed = format != StorageFormat::Floating;
    if packed && (shape.len() != 3 || !shape[2].is_multiple_of(block_size)) {
        return Err(import_error(
            specification,
            format!("is block-scaled but its leaf has shape {shape:?}, which is not a stack of block aligned matrices"),
        ));
    }
    let scaled = |divisor: usize| {
        let mut scaled = shape.clone();
        if let Some(extent) = scaled.last_mut() {
            *extent /= divisor;
        }
        scaled
    };
    if array.shape() != scaled(divisor) {
        return Err(import_error(
            specification,
            format!("has shape {:?} but shape {:?} is required", array.shape(), scaled(divisor)),
        ));
    }

    let sidecar = |suffix: &str, data_type: DataType, shape: Vec<usize>| {
        let sidecar_name = format!("{name}{suffix}");
        match checkpoint.array(&sidecar_name) {
            Some(sidecar) if sidecar.data_type() == data_type && sidecar.shape() == shape => Ok(()),
            Some(sidecar) => Err(CheckpointError::Import {
                message: format!(
                    "checkpoint array `{sidecar_name}` is `{}` {:?} but `{data_type}` {shape:?} is required",
                    sidecar.data_type(),
                    sidecar.shape(),
                ),
            }),
            None => Err(CheckpointError::Import { message: format!("checkpoint array `{sidecar_name}` is missing") }),
        }
    };
    match format {
        StorageFormat::Floating => {}
        StorageFormat::Nvfp4 => {
            sidecar("_scale", DataType::F8E4M3FN, scaled(NVFP4_BLOCK_SIZE))?;
            sidecar("_amax", DataType::F32, vec![shape[0]])?;
            sidecar("_outer_scale_denominator", DataType::F32, vec![1])?;
        }
        StorageFormat::Mxfp8 => sidecar("_scale", DataType::U8, scaled(MXFP8_BLOCK_SIZE))?,
    }
    Ok(format)
}

/// Returns the names of the checkpoint arrays that importing the leaf of `specification` reads.
fn required_arrays<C: Checkpoint>(
    checkpoint: &C,
    specification: &ArraySpecification,
) -> Result<Vec<String>, CheckpointError> {
    let name = specification.name();
    let suffixes: &[&str] = match storage_format(checkpoint, specification)? {
        StorageFormat::Floating => &[""],
        StorageFormat::Nvfp4 => &["", "_scale", "_amax", "_outer_scale_denominator"],
        StorageFormat::Mxfp8 => &["", "_scale"],
    };
    Ok(suffixes.iter().map(|suffix| format!("{name}{suffix}")).collect())
}

/// Reads the leaf of `specification` from `checkpoint` as an `f32` array, decoding packed projections.
fn read_array<C: Checkpoint>(checkpoint: &C, specification: &ArraySpecification) -> Result<Array, CheckpointError> {
    let name = specification.name();
    let format = storage_format(checkpoint, specification)?;
    if format == StorageFormat::Floating {
        return Ok(checkpoint.read(name)?.converted_to(DataType::F32)?);
    }

    let quantization_error =
        |error: CodecError| CheckpointError::InvalidArray { name: name.to_string(), message: error.to_string() };
    let shape = specification.r#type().static_shape().map(|shape| shape.dimensions().to_vec()).unwrap_or_default();
    let scales_name = format!("{name}_scale");
    let outer_scales = match format {
        StorageFormat::Nvfp4 => {
            let amax = checkpoint.read(&format!("{name}_amax"))?.elements::<f32>()?;
            let denominator = checkpoint.read(&format!("{name}_outer_scale_denominator"))?.elements::<f32>()?[0];
            amax.into_iter()
                .map(|amax| {
                    Nvfp4OuterScale::for_weights(amax, denominator).map(|outer_scale| Some(outer_scale.scale()))
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(quantization_error)?
        }
        _ => vec![None; shape[0]],
    };

    // One expert is expanded at a time, so the only large allocation is the leaf that is being assembled.
    let mut bytes = Vec::with_capacity(shape.iter().product::<usize>() * size_of::<f32>());
    for (expert, outer_scale) in outer_scales.into_iter().enumerate() {
        let elements = checkpoint.read_leading(name, expert..expert + 1)?;
        let scales = checkpoint.read_leading(&scales_name, expert..expert + 1)?;
        let values = match outer_scale {
            Some(outer_scale) => decode_nvfp4(elements.storage_bytes(), scales.storage_bytes(), outer_scale),
            None => decode_mxfp8(elements.storage_bytes(), scales.storage_bytes()),
        }
        .map_err(quantization_error)?;
        bytes.extend(values.into_iter().flat_map(f32::to_ne_bytes));
    }
    Ok(Array::new(ArrayType::new_static(DataType::F32, shape), bytes)?)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::ops::Range;

    use pretty_assertions::assert_eq;
    use ryft_checkpoints::ArrayDescriptor;
    use ryft_checkpoints::codecs::{Nvfp4ScaleSelection, encode_mxfp8, encode_nvfp4};

    use crate::inkling2::decoder::tests::{test_configuration, test_decoder};

    use super::*;

    /// [`Checkpoint`] that holds its arrays in memory, in the order in which they are provided.
    struct MemoryCheckpoint {
        /// Descriptors of the arrays, whose byte ranges follow each other.
        descriptors: Vec<ArrayDescriptor>,

        /// Contents of the arrays.
        contents: Vec<Vec<u8>>,
    }

    impl Checkpoint for MemoryCheckpoint {
        fn arrays(&self) -> &[ArrayDescriptor] {
            &self.descriptors
        }

        fn array(&self, name: &str) -> Option<&ArrayDescriptor> {
            self.descriptors.iter().find(|descriptor| descriptor.name() == name)
        }

        fn read_bytes(&self, name: &str, byte_range: Range<u64>) -> Result<Vec<u8>, CheckpointError> {
            let index = self
                .descriptors
                .iter()
                .position(|descriptor| descriptor.name() == name)
                .ok_or_else(|| CheckpointError::UnknownArray { name: name.to_string() })?;
            Ok(self.contents[index][byte_range.start as usize..byte_range.end as usize].to_vec())
        }
    }

    /// Returns an in-memory checkpoint that holds `arrays`.
    fn checkpoint(arrays: &[(String, Array)]) -> MemoryCheckpoint {
        let mut offset = 0;
        let mut descriptors = Vec::with_capacity(arrays.len());
        let mut contents = Vec::with_capacity(arrays.len());
        for (name, array) in arrays {
            let bytes = array.storage_bytes().to_vec();
            let r#type = array.r#type();
            let shape = r#type.static_shape().unwrap().dimensions().to_vec();
            descriptors.push(ArrayDescriptor::new(
                name.clone(),
                r#type.data_type(),
                shape,
                offset..offset + bytes.len() as u64,
            ));
            offset += bytes.len() as u64;
            contents.push(bytes);
        }
        MemoryCheckpoint { descriptors, contents }
    }

    #[test]
    fn test_inkling2_importer_new() {
        let importer = Inkling2Importer::new(test_configuration());
        assert_eq!(importer.configuration(), &test_configuration());
    }

    #[test]
    fn test_inkling2_importer_validate() {
        let configuration = test_configuration();
        let importer = Inkling2Importer::new(configuration.clone());
        let mut arrays = export_inkling2_decoder(&configuration, &test_decoder(&configuration)).unwrap();
        assert_eq!(importer.validate(&checkpoint(&arrays)), Ok(()));

        // Vision arrays are accepted and left unread, and other unaccounted arrays are rejected.
        arrays.push(("model.visual.proj_out.weight".to_string(), Array::scalar(0.0f32).unwrap()));
        assert_eq!(importer.validate(&checkpoint(&arrays)), Ok(()));
        arrays.push(("model.llm.extra".to_string(), Array::scalar(0.0f32).unwrap()));
        assert_eq!(
            importer.validate(&checkpoint(&arrays)),
            Err(CheckpointError::Import {
                message: "checkpoint array `model.llm.extra` is not an array of the Inkling 2 decoder".to_string(),
            }),
        );

        // A missing array and an array with a wrong shape are reported by name.
        let arrays = export_inkling2_decoder(&configuration, &test_decoder(&configuration)).unwrap();
        assert_eq!(
            importer.validate(&checkpoint(&arrays[1..])),
            Err(CheckpointError::Import {
                message: "checkpoint array `model.llm.embed.weight` is missing".to_string(),
            }),
        );
        let mut reshaped = arrays.clone();
        reshaped[1].1 = Array::vector(vec![0.0f32; 3]).unwrap();
        assert_eq!(
            importer.validate(&checkpoint(&reshaped)),
            Err(CheckpointError::Import {
                message: "checkpoint array `model.llm.embed_norm.weight` has shape [3] but shape [32] is required"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_inkling_2_mini_manifest() {
        let manifest = inkling_2_mini_manifest();
        assert_eq!(manifest.repository(), "thinkingmachines/Inkling-2-Mini");
        assert_eq!(manifest.revision(), "ed40e7da625f594fc9c4990f5f66be24873c22d7");
        assert_eq!(manifest.arrays(), &[]);
        let file = manifest.file(INKLING_2_MINI_CHECKPOINT_FILE).unwrap();
        assert_eq!(file.byte_count(), 19_526_019_168);
        assert_eq!(file.sha256(), Some("65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1"));
    }

    #[test]
    fn test_inkling2_importer_import() {
        let configuration = test_configuration();
        let decoder = test_decoder(&configuration);
        let arrays = export_inkling2_decoder(&configuration, &decoder).unwrap();
        let mut paths = Vec::new();
        let imported = Inkling2Importer::new(configuration.clone()).import(&checkpoint(&arrays), |path, array| {
            paths.push(path.to_string());
            Ok(array)
        });
        assert_eq!(imported, Ok(decoder.clone()));
        assert_eq!(paths.len(), decoder.parameter_count());
        assert_eq!(&paths[..2], &["$.embedding".to_string(), "$.embedding_norm".to_string()]);

        // Narrower floating-point arrays are converted to `f32` masters.
        let narrow = arrays
            .iter()
            .map(|(name, array)| (name.clone(), array.converted_to(DataType::BF16).unwrap()))
            .collect::<Vec<_>>();
        let imported = Inkling2Importer::new(configuration).import(&checkpoint(&narrow), |_, array| Ok(array)).unwrap();
        assert_eq!(
            imported.parameters().next().unwrap(),
            &arrays[0].1.converted_to(DataType::BF16).unwrap().converted_to(DataType::F32).unwrap(),
        );
    }

    #[test]
    fn test_inkling2_importer_import_packed_experts() {
        // Packs the expert projections of one NVFP4 layer and of one MXFP8 layer the way a release stores them, and
        // checks that importing them yields exactly the values that the packed arrays represent.
        let configuration = test_configuration();
        let decoder = test_decoder(&configuration);
        let mut arrays = Vec::new();
        let mut expected = BTreeMap::new();
        for (name, array) in export_inkling2_decoder(&configuration, &decoder).unwrap() {
            let layer = name
                .strip_prefix("model.llm.layers.")
                .and_then(|rest| rest.split('.').next()?.parse::<usize>().ok());
            let (Some(layer), true) = (layer, name.contains(".mlp.experts.w")) else {
                arrays.push((name, array));
                continue;
            };
            let shape = array.r#type().static_shape().unwrap().dimensions().to_vec();
            let values = array.elements::<f32>().unwrap();
            let expert_size = shape[1] * shape[2];
            let nvfp4 = layer < configuration.nvfp4_layer_count();
            let (mut elements, mut scales, mut amaxes, mut decoded) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for expert in values.chunks_exact(expert_size) {
                if nvfp4 {
                    let amax = expert.iter().fold(0.0f32, |amax, value| amax.max(value.abs()));
                    let outer_scale = Nvfp4OuterScale::for_weights(amax, 2688.0).unwrap();
                    let (packed, block_scales) = encode_nvfp4(expert, outer_scale, Nvfp4ScaleSelection::Six).unwrap();
                    decoded.extend(decode_nvfp4(&packed, &block_scales, outer_scale.scale()).unwrap());
                    elements.extend(packed);
                    scales.extend(block_scales);
                    amaxes.push(amax);
                } else {
                    let (values, exponents) = encode_mxfp8(expert).unwrap();
                    decoded.extend(decode_mxfp8(&values, &exponents).unwrap());
                    elements.extend(values);
                    scales.extend(exponents);
                }
            }
            let array = |data_type, shape: Vec<usize>, bytes: Vec<u8>| {
                Array::new(ArrayType::new_static(data_type, shape), bytes).unwrap()
            };
            if nvfp4 {
                arrays.push((name.clone(), array(DataType::U8, vec![shape[0], shape[1], shape[2] / 2], elements)));
                arrays.push((
                    format!("{name}_scale"),
                    array(DataType::F8E4M3FN, vec![shape[0], shape[1], shape[2] / 16], scales),
                ));
                arrays.push((format!("{name}_amax"), Array::vector(amaxes).unwrap()));
                arrays.push((format!("{name}_outer_scale_denominator"), Array::vector(vec![2688.0f32]).unwrap()));
            } else {
                arrays.push((name.clone(), array(DataType::F8E4M3FN, shape.clone(), elements)));
                arrays.push((
                    format!("{name}_scale"),
                    array(DataType::U8, vec![shape[0], shape[1], shape[2] / 32], scales),
                ));
            }
            expected.insert(name, decoded);
        }

        let specification = Inkling2Decoder::specification(&configuration);
        let imported = Inkling2Importer::new(configuration).import(&checkpoint(&arrays), |_, array| Ok(array)).unwrap();
        let mut compared = 0;
        for (leaf, array) in specification.parameters().zip(imported.parameters()) {
            if let Some(expected) = expected.get(leaf.name()) {
                assert_eq!(&array.elements::<f32>().unwrap(), expected);
                compared += 1;
            }
        }
        assert_eq!(compared, 2 * 8);
    }

    #[test]
    fn test_export_inkling2_decoder() {
        let configuration = test_configuration();
        let decoder = test_decoder(&configuration);
        let arrays = export_inkling2_decoder(&configuration, &decoder).unwrap();
        assert_eq!(arrays.len(), decoder.parameter_count());
        assert_eq!(arrays[0].0, "model.llm.embed.weight");
        assert_eq!(&arrays[0].1, decoder.parameters().next().unwrap());

        // A decoder with another structure and a decoder whose leaves have another type are both rejected.
        assert!(matches!(
            export_inkling2_decoder(&Inkling2Configuration::reduced(), &decoder),
            Err(CheckpointError::Parameter(_)),
        ));
        let narrow = decoder.map_parameters(|leaf| leaf.converted_to(DataType::BF16).unwrap()).unwrap();
        assert_eq!(
            export_inkling2_decoder(&configuration, &narrow).map(|_| ()),
            Err(CheckpointError::InvalidArray {
                name: "model.llm.embed.weight".to_string(),
                message: "leaf must have type `f32[64, 32]` but has type `bf16[64, 32]`".to_string(),
            }),
        );
    }
}
