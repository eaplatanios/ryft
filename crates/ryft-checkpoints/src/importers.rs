//! Importers, which map the arrays of a [`Checkpoint`] that some other system wrote to the leaves of a model
//! parameter structure.
//!
//! Model specific code owns the knowledge about a released checkpoint (i.e., array names, storage formats, and
//! layouts). That mapping is an [`Importer`], which hands every imported host [`Array`] to a caller provided function
//! together with the [`ParameterPath`] of its leaf. The function decides where the leaf lives (e.g., it uploads the
//! array to an accelerator and drops the host copy), which keeps exactly one copy of every parameter alive. Importers
//! usually describe a model as a parameter structure of [`ArraySpecification`]s and stream it with
//! [`import_arrays`].

use ryft_core::{Array, Parameter, ParameterPath, Parameterized, ParameterizedFamily, Typed};

use crate::arrays::{ArraySpecification, Checkpoint};
use crate::errors::CheckpointError;

// TODO(eaplatanios): Review this module.

/// Mapping from the arrays of a [`Checkpoint`] to the leaves of a model parameter structure. An importer owns the
/// model specific knowledge about a checkpoint (i.e., array names, storage formats, and layouts), while the caller
/// owns the placement of the imported leaves.
pub trait Importer {
    /// Family of the parameter structure that this importer produces, instantiated with the leaf type of the caller
    /// (e.g., a model parameter struct that is generic over its [`Parameter`] type).
    type Parameters<P: Parameter>: Parameterized<P>;

    /// Imports the parameters of a model from `checkpoint`. Every leaf is read (and decoded, where the checkpoint
    /// stores it in a packed format) into a host [`Array`] and immediately handed to `place` together with the
    /// [`ParameterPath`] of its leaf, in the order of [`Parameterized::parameters`]. The array is dropped as soon as
    /// `place` returns unless `place` keeps it, so at most one imported leaf is held in host memory by the importer at
    /// any time.
    ///
    /// # Parameters
    ///
    ///   - `checkpoint`: Checkpoint to import from.
    ///   - `place`: Function that turns the host [`Array`] of one leaf into the leaf value of the caller.
    ///
    /// # Errors
    ///
    /// Returns a [`CheckpointError`] if the checkpoint does not contain exactly the arrays that this importer
    /// expects with the expected data types and shapes, if an array cannot be read or decoded, or if `place` fails.
    fn import<C: Checkpoint, P: Parameter, F: FnMut(&ParameterPath, Array) -> Result<P, CheckpointError>>(
        &self,
        checkpoint: &C,
        place: F,
    ) -> Result<Self::Parameters<P>, CheckpointError>;
}

/// Imports every leaf of `specifications` in the order of [`Parameterized::parameters`]. Each leaf is produced by
/// `read`, checked against the type of its [`ArraySpecification`], and handed to `place` together with its
/// [`ParameterPath`]. This is the streaming loop that [`Importer`] implementations share. Their `read` function holds
/// the model specific knowledge about how an array is stored (e.g., it decodes a packed array), and `place` is the
/// function of the caller that decides where the leaf lives.
///
/// # Parameters
///
///   - `specifications`: Parameter structure that describes every leaf to import.
///   - `read`: Function that reads, and decodes or converts where necessary, the host [`Array`] of one leaf.
///   - `place`: Function that turns the host [`Array`] of one leaf into the leaf value of the caller.
///
/// # Errors
///
/// Returns the first error of `read` or `place`, and [`CheckpointError::InvalidArray`] if `read` returns an array
/// whose type differs from the specified one.
pub fn import_arrays<S, P, R, F>(specifications: S, mut read: R, mut place: F) -> Result<S::To<P>, CheckpointError>
where
    S: Parameterized<ArraySpecification>,
    S::Family: ParameterizedFamily<P>,
    P: Parameter,
    R: FnMut(&ArraySpecification) -> Result<Array, CheckpointError>,
    F: FnMut(&ParameterPath, Array) -> Result<P, CheckpointError>,
{
    specifications.try_map_named_parameters(|path, specification| {
        let array = read(&specification)?;
        if array.r#type().as_ref() != specification.r#type() {
            return Err(CheckpointError::InvalidArray {
                name: specification.name().to_string(),
                message: format!(
                    "leaf `{path}` must have type `{}` but the imported array has type `{}`",
                    specification.r#type(),
                    array.r#type(),
                ),
            });
        }
        place(path, array)
    })
}

// The only test reads arrays from a safetensors checkpoint.
#[cfg(all(test, feature = "safetensors"))]
mod tests {
    use pretty_assertions::assert_eq;
    use ryft_core::{ArrayType, DataType};

    use crate::arrays::tests::test_checkpoint;

    use super::*;

    #[test]
    fn test_import_arrays() {
        let checkpoint = test_checkpoint();
        let specifications = (
            ArraySpecification::new("scalar", ArrayType::scalar(DataType::U8)),
            vec![ArraySpecification::new("stack", ArrayType::new_static(DataType::F32, [3, 1, 2]))],
        );
        let mut placed = Vec::new();
        let imported = import_arrays(
            specifications.clone(),
            |specification| checkpoint.read(specification.name()),
            |path, array| {
                placed.push(path.to_string());
                Ok(array)
            },
        );
        assert_eq!(imported, Ok((checkpoint.read("scalar").unwrap(), vec![checkpoint.read("stack").unwrap()])));
        assert_eq!(placed, vec!["$.0".to_string(), "$.1[0]".to_string()]);

        // A leaf whose imported type differs from its specification is rejected before it is placed, and errors of
        // the reading and placing functions are returned unchanged.
        let mismatched =
            (ArraySpecification::new("scalar", ArrayType::scalar(DataType::F32)), Vec::<ArraySpecification>::new());
        assert_eq!(
            import_arrays(mismatched, |specification| checkpoint.read(specification.name()), |_, array| Ok(array),),
            Err(CheckpointError::InvalidArray {
                name: "scalar".to_string(),
                message: "leaf `$.0` must have type `f32[]` but the imported array has type `u8[]`".to_string(),
            }),
        );
        assert_eq!(
            import_arrays(
                specifications,
                |specification| checkpoint.read(specification.name()),
                |_, _| Err::<Array, _>(CheckpointError::Import { message: "no room".to_string() }),
            ),
            Err(CheckpointError::Import { message: "no room".to_string() }),
        );
    }
}
