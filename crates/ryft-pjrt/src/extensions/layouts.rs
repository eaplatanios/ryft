use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::iter::Peekable;
use std::marker::PhantomData;
use std::str::Chars;

use crate::protos::{DimensionSplit, Tile};
use crate::{Api, BufferType, Client, Error, Executable, Plugin, Topology, invoke_pjrt_api_error_fn, slice_from_c_api};

/// The PJRT layouts extension provides capabilities around custom on-device memory layouts for
/// [`Buffer`](crate::Buffer)s and [`Executable`]s. The extension is both optional for PJRT [`Plugin`]s and
/// _experimental_, meaning that incompatible changes may be introduced at any time, including changes that
/// break _Application Binary Interface (ABI)_ compatibility.
///
/// If this extension is provided by a PJRT [`Plugin`], then PJRT will assume that the compiler MLIR input programs
/// may contain `mhlo.layout_mode` attributes on program inputs and outputs, which should then be reflected by the
/// runtime functions on this extension.
///
/// Refer to [this file](https://github.com/openxla/xla/blob/main/xla/pjrt/layout_mode.h) for more information.
#[derive(Copy, Clone)]
pub struct LayoutsExtension {
    /// Handle that represents this [`LayoutsExtension`] in the PJRT C API.
    handle: *const ffi::PJRT_Layouts_Extension,

    /// Underlying PJRT [`Api`].
    api: Api,
}

impl LayoutsExtension {
    /// Constructs a new [`LayoutsExtension`] from the provided [`PJRT_Extension_Base`](crate::ffi::PJRT_Extension_Base)
    /// handle that came from a function in the PJRT C API if the type of that PJRT extension matches the PJRT layouts
    /// extension type.
    pub(crate) unsafe fn from_c_api(handle: *const crate::ffi::PJRT_Extension_Base, api: Api) -> Option<Self> {
        unsafe {
            if !handle.is_null() && (*handle).extension_type == crate::ffi::PJRT_Extension_Type_Layouts {
                Some(Self { handle: handle as *const _, api })
            } else {
                None
            }
        }
    }

    /// Returns the [`PJRT_Layouts_Extension`](ffi::PJRT_Layouts_Extension) that corresponds to this
    /// [`LayoutsExtension`] and which can be passed to functions in the PJRT C API.
    #[allow(clippy::wrong_self_convention)]
    pub(crate) unsafe fn to_c_api(&self) -> *const ffi::PJRT_Layouts_Extension {
        self.handle
    }

    /// Returns the underlying PJRT [`Api`].
    pub(crate) fn api(&self) -> Api {
        self.api
    }
}

unsafe impl Send for LayoutsExtension {}
unsafe impl Sync for LayoutsExtension {}

impl Client<'_> {
    /// Attempts to load the [`LayoutsExtension`] from this [`Client`] and returns [`Error::Unimplemented`]
    /// if it is not provided by the underlying [`Plugin`].
    pub fn layouts_extension(&self) -> Result<LayoutsExtension, Error> {
        self.api().layouts_extension()
    }
}

impl Plugin {
    /// Attempts to load the [`LayoutsExtension`] from this [`Plugin`] and returns [`Error::Unimplemented`]
    /// if it is not provided by this [`Plugin`].
    pub fn layouts_extension(&self) -> Result<LayoutsExtension, Error> {
        self.api().layouts_extension()
    }
}

impl Api {
    /// Attempts to load the [`LayoutsExtension`] from this [`Api`] and returns [`Error::Unimplemented`]
    /// if it is not provided by the underlying [`Plugin`].
    pub(crate) fn layouts_extension(&self) -> Result<LayoutsExtension, Error> {
        unsafe {
            let mut extension = (*self.to_c_api()).extension_start;
            while !extension.is_null() {
                let layouts_extension = LayoutsExtension::from_c_api(extension, *self);
                if let Some(layouts_extension) = layouts_extension {
                    return Ok(layouts_extension);
                }
                extension = (*extension).next;
            }
            Err(Error::unimplemented("the layouts extension is not provided by the PJRT plugin"))
        }
    }
}

/// Represents a PJRT [`Memory`](crate::Memory) layout for a [`Buffer`](crate::Buffer).
pub struct Layout<'o> {
    /// Handle that represents this [`Layout`] in the PJRT C API.
    handle: *mut ffi::PJRT_Layouts_MemoryLayout,

    /// [`LayoutsExtension`] that was used to create this [`Layout`].
    extension: LayoutsExtension,

    /// Boolean flag indicating whether this [`Layout`] is borrowed or owned. This influences the behavior of
    /// [`Layout`]'s [`Drop`] implementation as it will only free the underlying memory if the topology is owned
    /// (i.e., `is_borrowed` is set to `false`).
    is_borrowed: bool,

    /// [`PhantomData`] used to track the lifetime of the owner of this [`Layout`].
    owner: PhantomData<&'o ()>,
}

impl Layout<'_> {
    /// Constructs a new [`Layout`] from the provided [`PJRT_Layouts_MemoryLayout`](ffi::PJRT_Layouts_MemoryLayout)
    /// handle that came from a function in the PJRT C API and records whether that handle is owned by this wrapper.
    pub(crate) unsafe fn from_c_api(
        handle: *mut ffi::PJRT_Layouts_MemoryLayout,
        extension: LayoutsExtension,
        is_borrowed: bool,
    ) -> Result<Self, Error> {
        if handle.is_null() {
            Err(Error::invalid_argument("the provided PJRT memory layout handle is a null pointer"))
        } else {
            Ok(Self { handle, extension, is_borrowed, owner: PhantomData })
        }
    }

    /// Returns the [`PJRT_Layouts_MemoryLayout`](ffi::PJRT_Layouts_MemoryLayout) that corresponds to this [`Layout`]
    /// and which can be passed to functions in the PJRT C API.
    pub(crate) unsafe fn to_c_api(&self) -> *mut ffi::PJRT_Layouts_MemoryLayout {
        self.handle
    }

    /// Serializes this [`Layout`] into a Protobuf message.
    pub fn proto(&self) -> Result<crate::protos::Layout, Error> {
        // It would be nice to be able to get this directly without having to go through [`Layout::serialize`] first,
        // but unfortunately, the PJRT C API does not provide the necessary hooks for doing that.
        self.serialize()?.proto()
    }

    /// Serializes this [`Layout`] into a [`SerializedLayout`].
    pub fn serialize(&self) -> Result<SerializedLayout, Error> {
        use ffi::PJRT_Layouts_MemoryLayout_Serialize_Args;
        invoke_pjrt_api_error_fn!(
            @extension ffi::PJRT_Layouts_Extension => self.extension,
            PJRT_Layouts_MemoryLayout_Serialize,
            { layout = self.to_c_api() },
            { serialized_bytes, serialized_bytes_size, serialized_layout, serialized_layout_deleter },
        )
        .map(|(serialized_bytes, serialized_bytes_size, serialized_layout, serialized_layout_deleter)| {
            SerializedLayout {
                handle: serialized_layout,
                deleter: serialized_layout_deleter,
                data: serialized_bytes,
                data_size: serialized_bytes_size,
            }
        })
    }
}

impl PartialEq for Layout<'_> {
    fn eq(&self, other: &Self) -> bool {
        // This implementation is quite inefficient, but unfortunately, the C API does not provide a better way.
        let self_serialized = self.serialize();
        let other_serialized = other.serialize();
        self_serialized.is_ok() && other_serialized.is_ok() && self_serialized.unwrap() == other_serialized.unwrap()
    }
}

impl Eq for Layout<'_> {}

impl Display for Layout<'_> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.serialize().unwrap(), formatter)
    }
}

impl Debug for Layout<'_> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "Layout[{self}]")
    }
}

impl Drop for Layout<'_> {
    fn drop(&mut self) {
        if !self.is_borrowed {
            use ffi::PJRT_Layouts_MemoryLayout_Destroy_Args;
            invoke_pjrt_api_error_fn!(
                @extension ffi::PJRT_Layouts_Extension => self.extension,
                PJRT_Layouts_MemoryLayout_Destroy,
                { layout = self.to_c_api() },
            )
            .expect("failed to destroy PJRT memory layout");
        }
    }
}

impl<'s> Client<'s> {
    /// Returns the default memory [`Layout`] that this [`Client`] will use for buffers
    /// with the provided `element_type` and `dimensions`.
    pub fn default_layout<D: AsRef<[u64]>>(
        &'_ self,
        element_type: BufferType,
        dimensions: D,
    ) -> Result<Layout<'_>, Error> {
        use ffi::PJRT_Layouts_PJRT_Client_GetDefaultLayout_Args;
        let extension = self.layouts_extension()?;
        invoke_pjrt_api_error_fn!(
            @extension ffi::PJRT_Layouts_Extension => extension,
            PJRT_Layouts_PJRT_Client_GetDefaultLayout,
            {
                client = self.to_c_api(),
                element_type = element_type.to_c_api(),
                dims = dimensions.as_ref().as_ptr() as *const i64,
                num_dims = dimensions.as_ref().len(),
            },
            { layout },
        )
        .and_then(|handle| unsafe { Layout::from_c_api(handle, extension, false) })
    }
}

impl<'o> Topology<'o> {
    /// Returns the default memory [`Layout`] that will be used on this [`Topology`] for buffers
    /// with the provided `element_type` and `dimensions`.
    pub fn default_layout<D: AsRef<[u64]>>(
        &'_ self,
        element_type: BufferType,
        dimensions: D,
    ) -> Result<Layout<'_>, Error> {
        use ffi::PJRT_Layouts_PJRT_Topology_GetDefaultLayout_Args;
        let extension = self.api().layouts_extension()?;
        invoke_pjrt_api_error_fn!(
            @extension ffi::PJRT_Layouts_Extension => extension,
            PJRT_Layouts_PJRT_Topology_GetDefaultLayout,
            {
                topology_description = self.to_c_api(),
                element_type = element_type.to_c_api(),
                dims = dimensions.as_ref().as_ptr() as *const i64,
                num_dims = dimensions.as_ref().len(),
            },
            { layout },
        )
        .and_then(|handle| unsafe { Layout::from_c_api(handle, extension, false) })
    }
}

impl Executable {
    /// Returns the memory [`Layout`]s of the inputs of this [`Executable`].
    pub fn input_layouts(&'_ self) -> Result<Vec<Layout<'_>>, Error> {
        use ffi::PJRT_Layouts_PJRT_Executable_GetParameterLayouts_Args;
        let extension = self.api().layouts_extension()?;
        invoke_pjrt_api_error_fn!(
            @extension ffi::PJRT_Layouts_Extension => extension,
            PJRT_Layouts_PJRT_Executable_GetParameterLayouts,
            { executable = self.to_c_api() },
            { num_parameters, layouts },
        )
        .and_then(|(parameter_count, parameter_layouts)| {
            unsafe { slice_from_c_api(parameter_layouts, parameter_count) }
                .iter()
                .copied()
                .map(|handle| unsafe { Layout::from_c_api(handle, extension, true) })
                .collect::<Result<Vec<_>, _>>()
        })
    }

    /// Returns the memory [`Layout`]s of the outputs of this [`Executable`].
    pub fn output_layouts(&'_ self) -> Result<Vec<Layout<'_>>, Error> {
        use ffi::PJRT_Layouts_PJRT_Executable_GetOutputLayouts_Args;
        let extension = self.api().layouts_extension()?;
        invoke_pjrt_api_error_fn!(
            @extension ffi::PJRT_Layouts_Extension => extension,
            PJRT_Layouts_PJRT_Executable_GetOutputLayouts,
            { executable = self.to_c_api() },
            { num_outputs, layouts },
        )
        .and_then(|(output_count, output_layouts)| {
            unsafe { slice_from_c_api(output_layouts, output_count) }
                .iter()
                .copied()
                .map(|handle| unsafe { Layout::from_c_api(handle, extension, true) })
                .collect::<Result<Vec<_>, _>>()
        })
    }
}

/// Serialized [`Layout`].
pub struct SerializedLayout {
    /// Handle that represents this [`SerializedLayout`] in the PJRT C API.
    handle: *mut ffi::PJRT_Layouts_SerializedLayout,

    /// Optional function that must be called to free the underlying memory when dropping this instance.
    deleter: Option<unsafe extern "C" fn(topology: *mut ffi::PJRT_Layouts_SerializedLayout)>,

    /// Pointer to the underlying bytes of this [`SerializedLayout`].
    data: *const std::ffi::c_char,

    /// Size (i.e., number of bytes) of this [`SerializedLayout`].
    data_size: usize,
}

impl SerializedLayout {
    /// Constructs a [`SerializedLayout`] from the provided rendered [`Layout`].
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(rendered_layout: &str) -> Self {
        Self {
            handle: std::ptr::null_mut(),
            deleter: None,
            data: rendered_layout.as_ptr() as *const std::ffi::c_char,
            data_size: rendered_layout.len(),
        }
    }

    /// Returns a pointer to the underlying bytes of this [`SerializedLayout`].
    pub fn data(&self) -> &[u8] {
        unsafe { slice_from_c_api(self.data as *const _, self.data_size) }
    }

    /// Constructs a [`SerializedLayout`] from the provided [`Layout`](crate::protos::Layout) Protobuf.
    pub fn from_proto(layout: crate::protos::Layout) -> Result<Self, Error> {
        unsafe extern "C" fn delete_boxed_data(handle: *mut ffi::PJRT_Layouts_SerializedLayout) {
            if !handle.is_null() {
                unsafe { drop(Box::from_raw(handle as *mut Vec<u8>)) };
            }
        }

        let minor_to_major = layout.minor_to_major.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(",");
        let mut properties = Vec::new();

        if !layout.tiles.is_empty() {
            let mut rendered_tiles = String::from("T");
            for tile in layout.tiles {
                let dimensions = tile
                    .dimensions
                    .into_iter()
                    .map(|dimension| {
                        if dimension == i64::MIN {
                            Ok(String::from("*"))
                        } else if dimension >= 0 {
                            Ok(dimension.to_string())
                        } else {
                            Err(Error::invalid_argument(format!(
                                "invalid tile dimension '{dimension}': expected non-negative value or '*'",
                            )))
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?
                    .join(",");
                rendered_tiles.push('(');
                rendered_tiles.push_str(&dimensions);
                rendered_tiles.push(')');
            }
            properties.push(rendered_tiles);
        }

        if let Some(alignment) = layout.alignment {
            properties.push(format!("L({alignment})"));
        }

        if let Some(element_size_in_bits) = layout.element_size_in_bits {
            properties.push(format!("E({element_size_in_bits})"));
        }

        if let Some(memory_space) = layout.memory_space {
            properties.push(format!("S({memory_space})"));
        }

        if !layout.splits.is_empty() {
            let mut rendered_splits = String::from("SC");
            for split in layout.splits {
                let split_indices = split
                    .split_indices
                    .into_iter()
                    .map(|split_index| split_index.to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                rendered_splits.push('(');
                rendered_splits.push_str(&format!("{}:{split_indices}", split.dimension));
                rendered_splits.push(')');
            }
            properties.push(rendered_splits);
        }

        if let Some(dynamic_shape_metadata_prefix_size_in_bytes) = layout.dynamic_shape_metadata_prefix_size_in_bytes {
            properties.push(format!("M({dynamic_shape_metadata_prefix_size_in_bytes})"));
        }

        let rendered_layout = if properties.is_empty() {
            format!("{{{minor_to_major}}}")
        } else {
            format!("{{{minor_to_major}:{}}}", properties.join(""))
        };

        let rendered_layout = Box::new(rendered_layout.into_bytes());
        let data = rendered_layout.as_ptr() as *const std::ffi::c_char;
        let data_size = rendered_layout.len();
        let handle = Box::into_raw(rendered_layout) as *mut ffi::PJRT_Layouts_SerializedLayout;

        Ok(Self { handle, deleter: Some(delete_boxed_data), data, data_size })
    }

    /// Returns the Protobuf message that corresponds to this [`SerializedLayout`].
    pub fn proto(&self) -> Result<crate::protos::Layout, Error> {
        // This implementation relies on this [`SerializedLayout`] being a string that matches the pattern
        // `{minor_to_major:properties}` where:
        //
        //   - `minor_to_major` is a comma-separated list of dimension indices.
        //   - `properties` is an optional, `:`-separated list of properties, each of which has one of the
        //     following forms:
        //     - `T(d1,d2,...)(d1,d2,...)`: tiles (each group represents one tile using `*` for combined dimensions).
        //     - `L(n)`: tail padding alignment as a number of elements.
        //     - `E(n)`: element size as a number of bits.
        //     - `S(n)`: memory space index.
        //     - `SC(dim:i1,i2,...)`: split configurations.
        //     - `M(n)`: dynamic shape metadata prefix number of bytes.

        /// Parses an `i64` value from a string. The provided `description` is used for error reporting.
        fn parse_i64(value: &str, description: &str) -> Result<i64, Error> {
            value
                .trim()
                .parse::<i64>()
                .map_err(|error| Error::invalid_argument(format!("invalid {description} '{value}': {error}")))
        }

        /// Parses a `Vec<i64>` value from a string. The provided `description` is used for error reporting.
        fn parse_vec_i64(value: &str, description: &str) -> Result<Vec<i64>, Error> {
            if value.trim().is_empty() {
                Ok(Vec::new())
            } else {
                value.split(',').map(|item| parse_i64(item, description)).collect()
            }
        }

        /// Parses and consumes one balanced parenthesized group from `characters`. The provided [`Peekable`] must be
        /// positioned at `'('`. The returned string contains only the inner content, with the outer parentheses removed.
        /// Nested parentheses are supported and preserved in the returned content. Returns an [`Error::InvalidArgument`]
        /// if the next character is not `'('` or if the input ends before a matching closing `')'` is found.
        fn parse_parenthesized(characters: &mut Peekable<Chars>, context: &str) -> Result<String, Error> {
            if characters.next() != Some('(') {
                return Err(Error::invalid_argument(format!("expected '(' while parsing {context}")));
            }

            let mut content = String::new();
            let mut depth = 1;
            while depth > 0 {
                match characters.next() {
                    Some('(') => {
                        depth += 1;
                        content.push('(');
                    }
                    Some(')') => {
                        depth -= 1;
                        if depth > 0 {
                            content.push(')');
                        }
                    }
                    Some(c) => content.push(c),
                    None => {
                        return Err(Error::invalid_argument(format!(
                            "unexpected end of string while parsing {context}",
                        )));
                    }
                }
            }

            Ok(content)
        }

        let mut layout = crate::protos::Layout::default();

        let rendered_layout = std::str::from_utf8(self.data()).map_err(|error| {
            Error::invalid_argument(format!("serialized PJRT layout is not valid UTF-8: {}", error))
        })?;
        let rendered_layout =
            rendered_layout.trim().strip_prefix('{').and_then(|s| s.strip_suffix('}')).ok_or_else(|| {
                Error::invalid_argument(format!("layout string must be enclosed in braces: {rendered_layout}"))
            })?;

        let (minor_to_major, properties) = rendered_layout.split_once(':').unwrap_or((rendered_layout, ""));
        layout.minor_to_major = parse_vec_i64(minor_to_major, "minor_to_major dimension")?;
        let mut characters = properties.chars().peekable();
        while let Some(property) = characters.next() {
            match property {
                'T' => {
                    while characters.peek() == Some(&'(') {
                        let dimensions = parse_parenthesized(&mut characters, "tile")?;
                        let dimensions = if dimensions.trim().is_empty() {
                            Vec::new()
                        } else {
                            dimensions
                                .split(',')
                                .map(|dimension| {
                                    let dimension = dimension.trim();
                                    if dimension == "*" { Ok(i64::MIN) } else { parse_i64(dimension, "tile dimension") }
                                })
                                .collect::<Result<Vec<_>, _>>()?
                        };
                        layout.tiles.push(Tile { dimensions });
                    }
                }
                'L' => {
                    let alignment = parse_parenthesized(&mut characters, "alignment")?;
                    layout.alignment = Some(parse_i64(&alignment, "alignment")?);
                }
                'E' => {
                    let element_size_in_bits = parse_parenthesized(&mut characters, "element size")?;
                    layout.element_size_in_bits = Some(parse_i64(&element_size_in_bits, "element size")?);
                }
                'S' if characters.peek() == Some(&'C') => {
                    characters.next();
                    let mut has_split_group = false;
                    while characters.peek() == Some(&'(') {
                        has_split_group = true;
                        let content = parse_parenthesized(&mut characters, "split config")?;
                        let (dimension, split_indices) = content.split_once(':').ok_or_else(|| {
                            Error::invalid_argument(format!("split config missing ':' separator: {content}"))
                        })?;
                        layout.splits.push(DimensionSplit {
                            dimension: parse_i64(dimension, "split config dimension")?,
                            split_indices: parse_vec_i64(split_indices, "split index")?,
                        });
                    }
                    if !has_split_group {
                        return Err(Error::invalid_argument("expected '(' while parsing split config"));
                    }
                }
                'S' => {
                    let memory_space = parse_parenthesized(&mut characters, "memory space")?;
                    layout.memory_space = Some(parse_i64(&memory_space, "memory space")?);
                }
                'M' => {
                    let dynamic_shape_metadata_prefix_size_in_bytes =
                        parse_parenthesized(&mut characters, "metadata prefix size")?;
                    layout.dynamic_shape_metadata_prefix_size_in_bytes =
                        Some(parse_i64(&dynamic_shape_metadata_prefix_size_in_bytes, "metadata prefix size")?);
                }
                'D' | '#' | 'P' => {
                    return Err(Error::invalid_argument("sparse layouts are not supported"));
                }
                _ if property.is_whitespace() => {
                    // We simply skip whitespace characters to match XLA's permissive parsing of layouts.
                }
                _ => {
                    return Err(Error::invalid_argument(format!("unsupported layout property '{property}'")));
                }
            }
        }

        Ok(layout)
    }
}

impl PartialEq for SerializedLayout {
    fn eq(&self, other: &Self) -> bool {
        self.data() == other.data()
    }
}

impl Eq for SerializedLayout {}

impl Hash for SerializedLayout {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.data().hash(state);
    }
}

impl Display for SerializedLayout {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", std::str::from_utf8(self.data()).unwrap())
    }
}

impl Debug for SerializedLayout {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "SerializedLayout[{self}]")
    }
}

unsafe impl Send for SerializedLayout {}
unsafe impl Sync for SerializedLayout {}

impl Drop for SerializedLayout {
    fn drop(&mut self) {
        if let Some(deleter) = self.deleter {
            unsafe { deleter(self.handle) };
        }
    }
}

#[allow(dead_code, non_camel_case_types, non_snake_case, non_upper_case_globals)]
pub(crate) mod ffi {
    use std::marker::{PhantomData, PhantomPinned};

    use crate::buffers::ffi::{PJRT_Buffer, PJRT_Buffer_Type};
    use crate::clients::ffi::PJRT_Client;
    use crate::errors::ffi::PJRT_Error;
    use crate::ffi::PJRT_Extension_Base;
    use crate::programs::ffi::PJRT_Executable;
    use crate::topologies::ffi::PJRT_TopologyDescription;

    pub const PJRT_API_LAYOUTS_EXTENSION_VERSION: usize = 4;

    // We represent opaque C types as structs with a particular structure that is following the convention
    // suggested in [the Rustonomicon](https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs).
    #[repr(C)]
    pub struct PJRT_Layouts_MemoryLayout {
        _data: [u8; 0],
        _marker: PhantomData<(*mut u8, PhantomPinned)>,
    }

    #[repr(C)]
    pub struct PJRT_Layouts_PJRT_Buffer_MemoryLayout_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub buffer: *mut PJRT_Buffer,
        pub layout: *mut PJRT_Layouts_MemoryLayout,
    }

    impl PJRT_Layouts_PJRT_Buffer_MemoryLayout_Args {
        pub fn new(buffer: *mut PJRT_Buffer) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                buffer,
                layout: std::ptr::null_mut(),
            }
        }
    }

    pub type PJRT_Layouts_PJRT_Buffer_MemoryLayout =
        unsafe extern "C" fn(args: *mut PJRT_Layouts_PJRT_Buffer_MemoryLayout_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Layouts_MemoryLayout_Destroy_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub layout: *mut PJRT_Layouts_MemoryLayout,
    }

    impl PJRT_Layouts_MemoryLayout_Destroy_Args {
        pub fn new(layout: *mut PJRT_Layouts_MemoryLayout) -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), layout }
        }
    }

    pub type PJRT_Layouts_MemoryLayout_Destroy =
        unsafe extern "C" fn(args: *mut PJRT_Layouts_MemoryLayout_Destroy_Args) -> *mut PJRT_Error;

    // We represent opaque C types as structs with a particular structure that is following the convention
    // suggested in [the Rustonomicon](https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs).
    #[repr(C)]
    pub struct PJRT_Layouts_SerializedLayout {
        _data: [u8; 0],
        _marker: PhantomData<(*mut u8, PhantomPinned)>,
    }

    #[repr(C)]
    pub struct PJRT_Layouts_MemoryLayout_Serialize_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub layout: *mut PJRT_Layouts_MemoryLayout,
        pub serialized_bytes: *const std::ffi::c_char,
        pub serialized_bytes_size: usize,
        pub serialized_layout: *mut PJRT_Layouts_SerializedLayout,
        pub serialized_layout_deleter: Option<unsafe extern "C" fn(s_layout: *mut PJRT_Layouts_SerializedLayout)>,
    }

    impl PJRT_Layouts_MemoryLayout_Serialize_Args {
        pub fn new(layout: *mut PJRT_Layouts_MemoryLayout) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                layout,
                serialized_bytes: std::ptr::null(),
                serialized_bytes_size: 0,
                serialized_layout: std::ptr::null_mut(),
                serialized_layout_deleter: None,
            }
        }
    }

    pub type PJRT_Layouts_MemoryLayout_Serialize =
        unsafe extern "C" fn(args: *mut PJRT_Layouts_MemoryLayout_Serialize_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Layouts_PJRT_Client_GetDefaultLayout_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub client: *mut PJRT_Client,
        pub element_type: PJRT_Buffer_Type,
        pub dims: *const i64,
        pub num_dims: usize,
        pub layout: *mut PJRT_Layouts_MemoryLayout,
    }

    impl PJRT_Layouts_PJRT_Client_GetDefaultLayout_Args {
        pub fn new(
            client: *mut PJRT_Client,
            element_type: PJRT_Buffer_Type,
            dims: *const i64,
            num_dims: usize,
        ) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                client,
                element_type,
                dims,
                num_dims,
                layout: std::ptr::null_mut(),
            }
        }
    }

    pub type PJRT_Layouts_PJRT_Client_GetDefaultLayout =
        unsafe extern "C" fn(args: *mut PJRT_Layouts_PJRT_Client_GetDefaultLayout_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Layouts_PJRT_Topology_GetDefaultLayout_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub topology_description: *mut PJRT_TopologyDescription,
        pub element_type: PJRT_Buffer_Type,
        pub dims: *const i64,
        pub num_dims: usize,
        pub layout: *mut PJRT_Layouts_MemoryLayout,
    }

    impl PJRT_Layouts_PJRT_Topology_GetDefaultLayout_Args {
        pub fn new(
            topology_description: *mut PJRT_TopologyDescription,
            element_type: PJRT_Buffer_Type,
            dims: *const i64,
            num_dims: usize,
        ) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                topology_description,
                element_type,
                dims,
                num_dims,
                layout: std::ptr::null_mut(),
            }
        }
    }

    pub type PJRT_Layouts_PJRT_Topology_GetDefaultLayout =
        unsafe extern "C" fn(args: *mut PJRT_Layouts_PJRT_Topology_GetDefaultLayout_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Layouts_PJRT_Executable_GetParameterLayouts_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub executable: *mut PJRT_Executable,
        pub num_parameters: usize,
        pub layouts: *mut *mut PJRT_Layouts_MemoryLayout,
    }

    impl PJRT_Layouts_PJRT_Executable_GetParameterLayouts_Args {
        pub fn new(executable: *mut PJRT_Executable) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                executable,
                num_parameters: 0,
                layouts: std::ptr::null_mut(),
            }
        }
    }

    pub type PJRT_Layouts_PJRT_Executable_GetParameterLayouts =
        unsafe extern "C" fn(args: *mut PJRT_Layouts_PJRT_Executable_GetParameterLayouts_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Layouts_PJRT_Executable_GetOutputLayouts_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub executable: *mut PJRT_Executable,
        pub num_outputs: usize,
        pub layouts: *mut *mut PJRT_Layouts_MemoryLayout,
    }

    impl PJRT_Layouts_PJRT_Executable_GetOutputLayouts_Args {
        pub fn new(executable: *mut PJRT_Executable) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                executable,
                num_outputs: 0,
                layouts: std::ptr::null_mut(),
            }
        }
    }

    pub type PJRT_Layouts_PJRT_Executable_GetOutputLayouts =
        unsafe extern "C" fn(args: *mut PJRT_Layouts_PJRT_Executable_GetOutputLayouts_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Layouts_Extension {
        pub base: PJRT_Extension_Base,
        pub PJRT_Layouts_MemoryLayout_Destroy: Option<PJRT_Layouts_MemoryLayout_Destroy>,
        pub PJRT_Layouts_MemoryLayout_Serialize: Option<PJRT_Layouts_MemoryLayout_Serialize>,
        pub PJRT_Layouts_PJRT_Client_GetDefaultLayout: Option<PJRT_Layouts_PJRT_Client_GetDefaultLayout>,
        pub PJRT_Layouts_PJRT_Buffer_MemoryLayout: Option<PJRT_Layouts_PJRT_Buffer_MemoryLayout>,
        pub PJRT_Layouts_PJRT_Topology_GetDefaultLayout: Option<PJRT_Layouts_PJRT_Topology_GetDefaultLayout>,
        pub PJRT_Layouts_PJRT_Executable_GetOutputLayouts: Option<PJRT_Layouts_PJRT_Executable_GetOutputLayouts>,
        pub PJRT_Layouts_PJRT_Executable_GetParameterLayouts: Option<PJRT_Layouts_PJRT_Executable_GetParameterLayouts>,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use indoc::indoc;

    use crate::protos::{CompilationOptions, ExecutableCompilationOptions, Precision};
    use crate::tests::{test_cpu_client, test_cpu_plugin};
    use crate::{BufferType, Program};

    use super::SerializedLayout;

    #[test]
    fn test_layouts_extension() {
        assert!(test_cpu_plugin().layouts_extension().is_ok());
        assert!(test_cpu_client().layouts_extension().is_ok());
    }

    #[test]
    fn test_layout() {
        let client = test_cpu_client();
        let topology = client.topology().unwrap();
        let layout = client.default_layout(BufferType::F32, [2, 3, 4]).unwrap();
        let serialized = layout.serialize().unwrap();
        assert_eq!(SerializedLayout::from_proto(serialized.proto().unwrap()).unwrap(), serialized);
        assert_eq!(format!("{layout}"), "{2,1,0}");
        assert_eq!(format!("{layout:?}"), "Layout[{2,1,0}]");
        assert_eq!(format!("{serialized}"), "{2,1,0}");
        assert_eq!(format!("{serialized:?}"), "SerializedLayout[{2,1,0}]");
        assert_eq!(layout, topology.default_layout(BufferType::F32, [2, 3, 4]).unwrap());
    }

    #[test]
    fn test_serialized_layout() {
        let layout = SerializedLayout::from_str("{1,0}").proto().unwrap();
        assert_eq!(layout.minor_to_major, vec![1, 0]);
        assert!(layout.tiles.is_empty());
        assert_eq!(layout.alignment, None);

        let layout = SerializedLayout::from_str("{}").proto().unwrap();
        assert!(layout.minor_to_major.is_empty());

        let layout = SerializedLayout::from_str("{2,1,0}").proto().unwrap();
        assert_eq!(layout.minor_to_major, vec![2, 1, 0]);

        let layout = SerializedLayout::from_str("{1,0:T(4,2)}").proto().unwrap();
        assert_eq!(layout.minor_to_major, vec![1, 0]);
        assert_eq!(layout.tiles.len(), 1);
        assert_eq!(layout.tiles[0].dimensions, vec![4, 2]);

        let layout = SerializedLayout::from_str("{2,1,0:T(4,2)(8)}").proto().unwrap();
        assert_eq!(layout.tiles.len(), 2);
        assert_eq!(layout.tiles[0].dimensions, vec![4, 2]);
        assert_eq!(layout.tiles[1].dimensions, vec![8]);

        let layout = SerializedLayout::from_str("{1,0:T(*,4)}").proto().unwrap();
        assert_eq!(layout.tiles[0].dimensions, vec![i64::MIN, 4]);

        let layout = SerializedLayout::from_str("{1,0:L(64)}").proto().unwrap();
        assert_eq!(layout.alignment, Some(64));

        let layout = SerializedLayout::from_str("{1,0:E(32)}").proto().unwrap();
        assert_eq!(layout.element_size_in_bits, Some(32));

        let layout = SerializedLayout::from_str("{1,0:S(1)}").proto().unwrap();
        assert_eq!(layout.memory_space, Some(1));

        let layout = SerializedLayout::from_str("{1,0:M(8)}").proto().unwrap();
        assert_eq!(layout.dynamic_shape_metadata_prefix_size_in_bytes, Some(8));

        let layout = SerializedLayout::from_str("{1,0:SC(0:256,512)}").proto().unwrap();
        assert_eq!(layout.splits.len(), 1);
        assert_eq!(layout.splits[0].dimension, 0);
        assert_eq!(layout.splits[0].split_indices, vec![256, 512]);

        let layout = SerializedLayout::from_str("{2,1,0:T(4,2)L(2)E(16)S(1)SC(1:512)M(4)}").proto().unwrap();
        assert_eq!(layout.minor_to_major, vec![2, 1, 0]);
        assert_eq!(layout.tiles.len(), 1);
        assert_eq!(layout.alignment, Some(2));
        assert_eq!(layout.element_size_in_bits, Some(16));
        assert_eq!(layout.memory_space, Some(1));
        assert_eq!(layout.splits.len(), 1);
        assert_eq!(layout.dynamic_shape_metadata_prefix_size_in_bytes, Some(4));

        // Test some invalid serialized layouts.
        assert!(SerializedLayout::from_str("1,0").proto().is_err());
        assert!(SerializedLayout::from_str("{a,b}").proto().is_err());
        assert!(SerializedLayout::from_str("{1,0").proto().is_err());
        assert!(SerializedLayout::from_str("{:D(X)}").proto().is_err());
        assert!(SerializedLayout::from_str("{:#(unknown)}").proto().is_err());
        assert!(SerializedLayout::from_str("{:P(not_a_shape)}").proto().is_err());
        assert!(matches!(
            SerializedLayout::from_str("{:D(S~,C,H+)#(u4)*(u8)P(s64[4,2])}").proto(),
            Err(crate::Error::InvalidArgument { message, .. }) if message == "sparse layouts are not supported",
        ));
        assert!(matches!(
            SerializedLayout::from_str(
                "{2,1,0:D(D,C+~,H+)#(u4)*(u8)P(bf16[?,4,?,2]{2,1,0:T(4,2)(*)L(2)E(2)S(2)SC(1:512)(0:)M(4)})}",
            ).proto(),
            Err(crate::Error::InvalidArgument { message, .. }) if message == "sparse layouts are not supported",
        ));

        // Test Protobuf round-tripping.
        let layout = crate::protos::Layout {
            minor_to_major: vec![2, 1, 0],
            tiles: vec![crate::protos::Tile { dimensions: vec![4, i64::MIN] }],
            alignment: Some(8),
            element_size_in_bits: Some(16),
            memory_space: Some(1),
            splits: vec![
                crate::protos::DimensionSplit { dimension: 1, split_indices: vec![256, 512] },
                crate::protos::DimensionSplit { dimension: 0, split_indices: vec![] },
            ],
            dynamic_shape_metadata_prefix_size_in_bytes: Some(4),
            ..Default::default()
        };
        let serialized = SerializedLayout::from_proto(layout.clone()).unwrap();
        assert_eq!(format!("{serialized}"), "{2,1,0:T(4,*)L(8)E(16)S(1)SC(1:256,512)(0:)M(4)}");
        assert_eq!(serialized.proto().unwrap(), layout);

        // Test using an invalid layout Protobuf message.
        assert!(matches!(
            SerializedLayout::from_proto(
                crate::protos::Layout {
                    tiles: vec![crate::protos::Tile { dimensions: vec![-3] }],
                    ..Default::default()
                },
            ),
            Err(crate::Error::InvalidArgument { message, .. })
                if message == "invalid tile dimension '-3': expected non-negative value or '*'",
        ));
    }

    #[test]
    fn test_executable_layouts() {
        let client = test_cpu_client();
        let program = Program::Mlir {
            bytecode: indoc! {"
                module {
                  func.func @main(%arg0: tensor<2x1xi32>, %arg1: tensor<2x1xi32>) -> tensor<2x1xi32> {
                    %0 = stablehlo.add %arg0, %arg1 : tensor<2x1xi32>
                    return %0 : tensor<2x1xi32>
                  }
                }
            "}
            .as_bytes()
            .to_vec(),
        };
        let compilation_options = CompilationOptions {
            argument_layouts: Vec::new(),
            parameter_is_tupled_arguments: false,
            executable_build_options: Some(ExecutableCompilationOptions {
                device_ordinal: -1,
                replica_count: 1,
                partition_count: 1,
                ..Default::default()
            }),
            compile_portable_executable: false,
            profile_version: 0,
            serialized_multi_slice_configuration: Vec::new(),
            environment_option_overrides: HashMap::new(),
            target_config: None,
            allow_in_place_mlir_modification: false,
            matrix_unit_operand_precision: Precision::Default as i32,
        };
        let executable = client.compile(&program, &compilation_options).unwrap();
        let executable = executable.executable().unwrap();
        let input_layouts = executable.input_layouts().unwrap();
        assert_eq!(input_layouts.len(), 2);
        for input_layout in input_layouts {
            assert_eq!(format!("{}", input_layout.serialize().unwrap()), "{1,0}");
        }
        let output_layouts = executable.output_layouts().unwrap();
        assert_eq!(output_layouts.len(), executable.output_count().unwrap());
        for output_layout in output_layouts {
            assert_eq!(format!("{}", output_layout.serialize().unwrap()), "{1,0}");
        }
    }
}
