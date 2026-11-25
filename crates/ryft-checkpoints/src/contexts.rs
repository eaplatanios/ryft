//! Contexts of saving and loading, mirroring the `Context` of the Orbax v1 API.
//!
//! A [`Context`] carries everything that configures how checkpoints are saved and loaded, apart from what is saved
//! and where. It owns the [`LeafHandler`] that converts the leaves of parameter structures to host arrays and groups
//! the remaining configuration into option groups, as Orbax does:
//!
//!   | Option group               | Orbax counterpart        | Configures                                         |
//!   |----------------------------|--------------------------|----------------------------------------------------|
//!   | [`ParameterizedOptions`]   | `PyTreeOptions`          | partial loading, renaming, data type conversion    |
//!   | [`ArrayOptions`]           | `ArrayOptions`           | the [`ArrayFormat`] of the arrays                  |
//!   | [`MultiprocessingOptions`] | `MultiprocessingOptions` | the role of this process among saving processes    |
//!   | [`Atomicity`]              | temporary path classes   | how checkpoints are finalized                      |
//!
//! Orbax activates a context for a block of code through a context variable, which the free functions of its API pick
//! up. Ryft has no ambient state, so every function that needs a context takes it as its last argument (e.g.,
//! [`save`](crate::saving::save) and [`load`](crate::loading::load)), and a
//! [`training::Checkpointer`](crate::training::Checkpointer) owns the context that it saves and loads with.

use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use ryft_core::ParameterPath;

use crate::formats::ArrayFormat;
use crate::handlers::parameterized::{ArrayLeafHandler, LeafHandler, ParameterizedHandler};
use crate::multiprocessing::MultiprocessingOptions;
use crate::paths::Atomicity;
#[cfg(feature = "safetensors")]
use crate::safetensors::SafetensorsFormat;

// TODO(eaplatanios): Review this module.

/// Options of saving and loading parameter structures, which mirror the `PyTreeOptions` of Orbax. All of the current
/// options apply to loading.
///
/// By default, loading is strict. Every leaf of the abstract state is loaded from the stored array that is named by
/// its own path and must have exactly the leaf type of the abstract state, and the checkpoint must not hold arrays
/// that the abstract state does not name. [`Self::with_partial_load`] ignores such arrays instead, which loads a part
/// of a saved structure. [`Self::with_names`] and [`Self::with_prefix`] load leaves from arrays with other names
/// (e.g., after a model was restructured, or to load one substructure of a saved structure), and
/// [`Self::with_data_type_conversion`] converts stored arrays whose data types differ from the leaf types of the
/// abstract state. Shapes must always match exactly, because padding and truncating arrays (i.e., the `strict=False`
/// loading of Orbax) is not supported.
#[derive(Clone, Default)]
pub struct ParameterizedOptions {
    /// Whether stored arrays that the abstract state does not name are ignored.
    partial_load: bool,

    /// Whether stored arrays whose data types differ from the leaf types of the abstract state are converted.
    convert_data_types: bool,

    /// Mapping from the path of every leaf of the abstract state to the name of the stored array that it is loaded
    /// from, if any.
    names: Option<Arc<NameMapping>>,
}

impl ParameterizedOptions {
    /// Creates new [`ParameterizedOptions`] that load strictly, from the stored arrays named by the leaf paths, and
    /// without data type conversion.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns these options with stored arrays that the abstract state does not name ignored when `partial_load` is
    /// `true`, which mirrors the `partial_load` option of the `PyTreeOptions.Loading` of Orbax.
    #[inline]
    pub fn with_partial_load(mut self, partial_load: bool) -> Self {
        self.partial_load = partial_load;
        self
    }

    /// Returns these options with stored arrays whose data types differ from the leaf types of the abstract state
    /// converted to those types when `convert` is `true`, which mirrors the `dtype` of the restore arguments of Orbax.
    /// Conversions follow [`ConvertElementType`](ryft_core::ConvertElementType) of host arrays.
    #[inline]
    pub fn with_data_type_conversion(mut self, convert: bool) -> Self {
        self.convert_data_types = convert;
        self
    }

    /// Returns these options with `names` mapping the path of every leaf of the abstract state to the name of the
    /// stored array to load it from, which mirrors the `transforms` of Orbax.
    #[inline]
    pub fn with_names<N: 'static + Send + Sync + Fn(&ParameterPath) -> String>(mut self, names: N) -> Self {
        self.names = Some(Arc::new(names));
        self
    }

    /// Returns these options with the abstract state loaded from the substructure of the saved structure at `prefix`,
    /// which maps the leaf path `$<rest>` to the stored name `<prefix><rest>` (e.g., the prefix `$.0` loads an abstract
    /// state of a single array from the first element of a saved pair). This replaces names set with
    /// [`Self::with_names`]. Loading a substructure ignores the rest of the saved structure only with
    /// [`Self::with_partial_load`].
    pub fn with_prefix(self, prefix: &str) -> Self {
        let prefix = prefix.to_string();
        self.with_names(move |path| {
            let path = path.to_string();
            format!("{prefix}{}", path.strip_prefix('$').unwrap_or(&path))
        })
    }

    /// Returns whether stored arrays that the abstract state does not name are ignored.
    #[inline]
    pub fn partial_load(&self) -> bool {
        self.partial_load
    }

    /// Returns whether stored arrays whose data types differ from the leaf types of the abstract state are converted.
    #[inline]
    pub fn convert_data_types(&self) -> bool {
        self.convert_data_types
    }

    /// Returns the name of the stored array that the leaf at `path` is loaded from.
    pub fn array_name(&self, path: &ParameterPath) -> String {
        match &self.names {
            Some(names) => names(path),
            None => path.to_string(),
        }
    }
}

impl Debug for ParameterizedOptions {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ParameterizedOptions")
            .field("partial_load", &self.partial_load)
            .field("convert_data_types", &self.convert_data_types)
            .field("renamed", &self.names.is_some())
            .finish()
    }
}

/// Options of the arrays of checkpoints, which mirror the `ArrayOptions` of Orbax and select the [`ArrayFormat`] that
/// stores them.
#[derive(Clone)]
pub struct ArrayOptions {
    /// Storage format of the arrays.
    format: Arc<dyn ArrayFormat>,
}

impl ArrayOptions {
    /// Creates new [`ArrayOptions`] that store arrays in `format`.
    #[inline]
    pub fn new(format: Arc<dyn ArrayFormat>) -> Self {
        Self { format }
    }

    /// Returns the storage format of the arrays.
    #[inline]
    pub fn format(&self) -> &Arc<dyn ArrayFormat> {
        &self.format
    }
}

impl Debug for ArrayOptions {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("ArrayOptions").field("format", &self.format.name()).finish()
    }
}

/// Configuration of saving and loading, which mirrors the `Context` of Orbax. Refer to the
/// [module documentation](self) for its option groups and how it is passed around.
///
/// A context is cheap to clone when its leaf handler is, because its array format and name mapping are shared.
/// [`Default::default`] creates the context for host arrays stored in one safetensors file, with the `safetensors`
/// feature.
#[derive(Clone)]
pub struct Context<L: LeafHandler = ArrayLeafHandler> {
    /// Conversion between the leaves of parameter structures and host arrays.
    leaf_handler: L,

    /// Options of saving and loading parameter structures.
    parameterized_options: ParameterizedOptions,

    /// Options of the arrays of checkpoints.
    array_options: ArrayOptions,

    /// Role of this process among the processes that save checkpoints together.
    multiprocessing_options: MultiprocessingOptions,

    /// Atomicity mode of saving, or [`None`] for the default mode of the file system of each checkpoint.
    atomicity: Option<Atomicity>,
}

impl<L: LeafHandler> Context<L> {
    /// Creates a new [`Context`] that converts leaves with `leaf_handler` and stores arrays in `array_format`, with
    /// default [`ParameterizedOptions`] and [`MultiprocessingOptions`], saving in the default [`Atomicity`] mode of the
    /// file system of each checkpoint.
    #[inline]
    pub fn new(leaf_handler: L, array_format: Arc<dyn ArrayFormat>) -> Self {
        Self {
            leaf_handler,
            parameterized_options: ParameterizedOptions::default(),
            array_options: ArrayOptions::new(array_format),
            multiprocessing_options: MultiprocessingOptions::default(),
            atomicity: None,
        }
    }

    /// Returns this context with the provided options of saving and loading parameter structures.
    #[inline]
    pub fn with_parameterized_options(mut self, parameterized_options: ParameterizedOptions) -> Self {
        self.parameterized_options = parameterized_options;
        self
    }

    /// Returns this context with the provided options of the arrays of checkpoints.
    #[inline]
    pub fn with_array_options(mut self, array_options: ArrayOptions) -> Self {
        self.array_options = array_options;
        self
    }

    /// Returns this context configured to save together with other processes in the role that
    /// `multiprocessing_options` describes. Refer to the [`multiprocessing`](crate::multiprocessing) module for the
    /// phases of a save across processes.
    #[inline]
    pub fn with_multiprocessing_options(mut self, multiprocessing_options: MultiprocessingOptions) -> Self {
        self.multiprocessing_options = multiprocessing_options;
        self
    }

    /// Returns this context configured to save in the `atomicity` mode on every file system. Saving in
    /// [`Atomicity::Rename`] mode on a file system that cannot rename atomically is not atomic.
    #[inline]
    pub fn with_atomicity(mut self, atomicity: Atomicity) -> Self {
        self.atomicity = Some(atomicity);
        self
    }

    /// Returns the conversion between the leaves of parameter structures and host arrays.
    #[inline]
    pub fn leaf_handler(&self) -> &L {
        &self.leaf_handler
    }

    /// Returns the options of saving and loading parameter structures.
    #[inline]
    pub fn parameterized_options(&self) -> &ParameterizedOptions {
        &self.parameterized_options
    }

    /// Returns the options of the arrays of checkpoints.
    #[inline]
    pub fn array_options(&self) -> &ArrayOptions {
        &self.array_options
    }

    /// Returns the role of this process among the processes that save checkpoints together.
    #[inline]
    pub fn multiprocessing_options(&self) -> &MultiprocessingOptions {
        &self.multiprocessing_options
    }

    /// Returns the configured atomicity mode, or [`None`] if the default mode of each file system is used.
    #[inline]
    pub fn atomicity(&self) -> Option<Atomicity> {
        self.atomicity
    }

    /// Returns the [`ParameterizedHandler`] that saves and loads parameter structures with the leaf handler, the array
    /// format, and the parameterized options of this context, which is the handler of the `state` checkpointable of
    /// [`save`](crate::saving::save) and [`load`](crate::loading::load).
    #[inline]
    pub fn parameterized_handler(&self) -> ParameterizedHandler<'_, L> {
        ParameterizedHandler::new(&self.leaf_handler, self.array_options.format().clone(), &self.parameterized_options)
    }
}

#[cfg(feature = "safetensors")]
impl Default for Context {
    fn default() -> Self {
        Self::new(ArrayLeafHandler, Arc::new(SafetensorsFormat))
    }
}

impl<L: LeafHandler + Debug> Debug for Context<L> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Context")
            .field("leaf_handler", &self.leaf_handler)
            .field("parameterized_options", &self.parameterized_options)
            .field("array_options", &self.array_options)
            .field("multiprocessing_options", &self.multiprocessing_options)
            .field("atomicity", &self.atomicity)
            .finish()
    }
}

/// Mapping from the path of a leaf of an abstract state to the name of the stored array that it is loaded from.
type NameMapping = dyn Fn(&ParameterPath) -> String + Send + Sync;

#[cfg(test)]
pub(crate) mod tests {
    use pretty_assertions::assert_eq;

    use crate::formats::tests::TestFormat;

    use super::*;

    /// Returns a context for host arrays stored in the [`TestFormat`], which every feature configuration provides.
    pub(crate) fn test_context() -> Context {
        Context::new(ArrayLeafHandler, Arc::new(TestFormat))
    }

    #[test]
    fn test_parameterized_options() {
        let options = ParameterizedOptions::new();
        assert!(!options.partial_load());
        assert!(!options.convert_data_types());
        assert_eq!(options.array_name(&ParameterPath::root().field("weight")), "$.weight");
        assert_eq!(
            format!("{options:?}"),
            "ParameterizedOptions { partial_load: false, convert_data_types: false, renamed: false }",
        );

        let options = options.with_partial_load(true).with_data_type_conversion(true).with_prefix("$.0");
        assert!(options.partial_load());
        assert!(options.convert_data_types());
        assert_eq!(options.array_name(&ParameterPath::root()), "$.0");
        assert_eq!(options.array_name(&ParameterPath::root().field("weight")), "$.0.weight");
        let options = options.with_names(|path| format!("model/{path}"));
        assert_eq!(options.array_name(&ParameterPath::root().field("weight")), "model/$.weight");
    }

    #[test]
    fn test_array_options() {
        let options = ArrayOptions::new(Arc::new(TestFormat));
        assert_eq!(options.format().name(), "test");
        assert_eq!(format!("{options:?}"), "ArrayOptions { format: \"test\" }");
    }

    #[test]
    fn test_context_new() {
        let context = test_context();
        assert_eq!(context.leaf_handler(), &ArrayLeafHandler);
        assert!(!context.parameterized_options().partial_load());
        assert_eq!(context.array_options().format().name(), "test");
        assert!(context.multiprocessing_options().is_primary());
        assert_eq!(context.atomicity(), None);
        assert_eq!(context.parameterized_handler().format().name(), "test");

        let context = context
            .with_parameterized_options(ParameterizedOptions::new().with_partial_load(true))
            .with_array_options(ArrayOptions::new(Arc::new(TestFormat)))
            .with_multiprocessing_options(MultiprocessingOptions::new(1, 2).unwrap())
            .with_atomicity(Atomicity::CommitFile);
        assert!(context.parameterized_options().partial_load());
        assert!(context.parameterized_handler().options().partial_load());
        assert!(!context.multiprocessing_options().is_primary());
        assert_eq!(context.atomicity(), Some(Atomicity::CommitFile));
    }

    #[cfg(feature = "safetensors")]
    #[test]
    fn test_context_default() {
        assert_eq!(Context::default().array_options().format().name(), "safetensors");
    }
}
