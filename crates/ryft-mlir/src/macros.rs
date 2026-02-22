/// Generates trait implementations for MLIR subtype wrapper structs (e.g., [`Type`](crate::Type),
/// [`Attribute`](crate::Attribute), etc.).
///
/// This macro generates implementations for the super-type trait (e.g., [`Type`](crate::Type) for
/// [`IntegerTypeRef`](crate::IntegerTypeRef)), as well as standard trait implementations for [`PartialEq`], [`Eq`],
/// [`Display`](std::fmt::Display), and [`Debug`](std::fmt::Debug).
///
/// # Parameters
///
///   - `subtype`: Name of the subtype struct (e.g., `IntegerTypeRef`).
///   - `lifetimes`: Lifetime parameters for the subtype (e.g., `'c, 't`).
///   - `type`: Name of the super-type trait (e.g., `Type`).
///   - `mlir_type`: MLIR C API type name without the `Mlir` prefix (e.g., `Type` for `MlirType`).
///   - `mlir_subtype`: MLIR C API subtype name (e.g., `IntegerType`).
///   - `mlir_prefix`: Optional prefix for MLIR C API functions (e.g., `stablehlo`) that defaults to `mlir`.
///
/// # Example
///
/// ```ignore
/// mlir_subtype_trait_impls!(
///     IntegerTypeRef<'c, 't> as Type,
///     mlir_type = Type,
///     mlir_subtype = Integer,
/// );
/// ```
///
/// For this example, this macro will generate implementations for the following traits:
///   - [`Type`](crate::Type),
///   - [`PartialEq`] using `mlirTypeEqual`,
///   - [`Eq`],
///   - [`Display`](std::fmt::Display) using `mlirTypePrint`, and
///   - [`Debug`](std::fmt::Debug) wrapping the [`Display`](std::fmt::Display) output.
#[macro_export]
macro_rules! mlir_subtype_trait_impls {
    (
        $subtype:ident <$($lifetimes:tt),*> as $type:ident,
        mlir_type = $mlir_type:ident,
        mlir_subtype = $mlir_subtype:ident $(,)*
    ) => {
        mlir_subtype_trait_impls!(
            $subtype <$($lifetimes),*> as $type,
            mlir_type = $mlir_type,
            mlir_subtype = $mlir_subtype,
            mlir_prefix = mlir,
        );
    };
    (
        $subtype:ident <$($lifetimes:tt),*> as $type:ident,
        mlir_type = $mlir_type:ident,
        mlir_subtype = $mlir_subtype:ident,
        mlir_prefix = $mlir_prefix:ident $(,)*
    ) => {
        paste::paste! {
            impl<$($lifetimes),*> $type<$($lifetimes),*> for $subtype<$($lifetimes),*> {
                unsafe fn from_c_api(
                    handle: ryft_xla_sys::bindings::[<Mlir $mlir_type>],
                    context: &'c $crate::Context<'t>,
                ) -> Option<Self> {
                    if !handle.ptr.is_null() && unsafe {
                        ryft_xla_sys::bindings::[<$mlir_prefix $mlir_type IsA $mlir_subtype>](handle)
                    } {
                        Some(Self { handle, context })
                    } else {
                        None
                    }
                }

                unsafe fn to_c_api(&self) -> ryft_xla_sys::bindings::[<Mlir $mlir_type>] {
                    self.handle
                }

                fn context(&self) -> &'c $crate::Context<'t> {
                    &self.context
                }
            }

            $crate::mlir_subtype_trait_impls!($subtype<$($lifetimes),*> as $type, mlir_type = $mlir_type);
        }
    };
    ($subtype:ident <$($lifetimes:tt),*> as $type:ident, mlir_type = $mlir_type:ident $(,)*) => {
        paste::paste! {
            impl <$($lifetimes),*, T: $type<$($lifetimes),*>> PartialEq<T> for $subtype <$($lifetimes),*> {
                fn eq(&self, other: &T) -> bool {
                    unsafe { ryft_xla_sys::bindings::[<mlir $mlir_type Equal>](self.to_c_api(), other.to_c_api()) }
                }
            }

            impl <$($lifetimes),*> Eq for $subtype <$($lifetimes),*> {}

            impl <$($lifetimes),*> std::fmt::Display for $subtype <$($lifetimes),*> {
                fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                    let mut data = (formatter, Ok(()));
                    unsafe {
                        ryft_xla_sys::bindings::[<mlir $mlir_type Print>](
                            self.to_c_api(),
                            Some($crate::support::write_to_formatter_callback),
                            &mut data as *mut _ as *mut std::ffi::c_void,
                        );
                    }
                    data.1
                }
            }

            impl <$($lifetimes),*> std::fmt::Debug for $subtype <$($lifetimes),*> {
                fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    write!(formatter, "{}[{}]", stringify!($subtype), self.to_string())
                }
            }
        }
    };
}

/// Generates an enum type and corresponding MLIR [`Attribute`](crate::Attribute) wrapper for string-based enumeration
/// attributes in MLIR dialects.
///
/// This macro creates:
///   - a Rust enum type representing the enumeration values,
///   - a [`TryFrom<&str>`] implementation for parsing string values,
///   - an attribute wrapper struct,
///   - an accessor method for retrieving the enum value that corresponds to the attribute value, and
///   - a [`Context`](crate::Context) method to construct new instances of that attribute.
///
/// # Parameters
///
///   - `rust_name`: Name of the Rust enum type (e.g., `Comparison`).
///   - `mlir_name`: MLIR attribute type name used in the C API functions (e.g., `Comparison`).
///   - `description`: Human-readable description of what the attribute represents (e.g., "comparison type").
///   - `variants`: Mapping from Rust enum variants to their MLIR string representations (e.g., `Eq => "EQ"`).
///   - `rust_prefix`: Optional prefix for the [`Context`](crate::Context) constructor method name.
///   - `mlir_prefix`: Optional prefix for MLIR C API functions (e.g., `stablehlo`) that defaults to `mlir`.
///   - `mlir_dialect_handle_constructor`: Optional dialect handle constructor to ensure that the dialect is loaded.
///
/// # Example
///
/// ```ignore
/// mlir_enum_attribute!(
///     rust_name = ComparisonDirection,
///     mlir_name = ComparisonDirection,
///     description = "StableHLO comparison direction",
///     variants = {
///         Equal => "EQ",
///         NotEqual => "NE",
///         GreaterThanOrEqual => "GE",
///         GreaterThan => "GT",
///         LessThanOrEqual => "LE",
///         LessThan => "LT",
///     },
///     rust_prefix = stable_hlo,
///     mlir_prefix = stablehlo,
///     mlir_dialect_handle_constructor = stable_hlo,
/// );
/// ```
///
/// For this example, this macro will generate the following:
///   - Rust enum called `ComparisonDirection` with variants `Equal`, `NotEqual`, etc.
///   - [`TryFrom<&str>`] implementation for the `ComparisonDirection` enum.
///   - Rust struct called `ComparisonDirectionAttributeRef<'c, 't>` that wraps an MLIR string attribute.
///   - A `value(&self) -> ComparisonDirection` function for that struct that enables extracting the enum value.
///   - A constructor in [`Context<'t>`](crate::Context) with the following signature:
///     `fn comparison_direction<'c>(&'c self, value: ComparisonDirection) -> ComparisonAttributeRef<'c, 't>`.
///   - `ComparisonDirectionAttributeRef` implementations for the following traits:
///     [`Attribute`](crate::Attribute), [`PartialEq`], [`Eq`], [`Display`](std::fmt::Display),
///     and [`Debug`](std::fmt::Debug).
#[macro_export]
macro_rules! mlir_enum_attribute {
    (
        rust_name = $rust_name:ident,
        mlir_name = $mlir_name:ident,
        { $($rust_variant:ident => $mlir_variant:literal),+ $(,)* },
        description = $description:literal
        $(,rust_prefix = $rust_prefix:ident)?
        $(,)*
    ) => {
        mlir_enum_attribute!(
            rust_name = $rust_name,
            mlir_name = $mlir_name,
            description = $description,
            variants = { $($rust_variant => $mlir_variant),+ },
            $(rust_prefix = $rust_prefix:ident,)?
            mlir_prefix = mlir,
        );
    };
    (
        rust_name = $rust_name:ident,
        mlir_name = $mlir_name:ident,
        description = $description:literal,
        variants = { $($rust_variant:ident => $mlir_variant:literal),+ $(,)* },
        $(rust_prefix = $rust_prefix:ident,)?
        mlir_prefix = $mlir_prefix:ident
        $(, mlir_dialect_handle_constructor = $mlir_dialect_handle_constructor:ident)? $(,)*
    ) => {
        paste::paste! {
            #[doc = "Represents a"]
            #[doc = $description]
            #[doc = "in MLIR."]
            #[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
            pub enum $rust_name {
                $($rust_variant,)+
            }

            impl<'s> TryFrom<&'s str> for $rust_name {
                type Error = String;

                fn try_from(value: &'s str) -> Result<Self, Self::Error> {
                    match value.as_ref() {
                        $($mlir_variant => Ok(Self::$rust_variant),)+
                        _ => Err(format!("'{}' is not a valid {}", value, $description)),
                    }
                }
            }

            #[doc = "MLIR [`Attribute`] that holds a"]
            #[doc = $description]
            #[doc = "reference."]
            #[derive(Copy, Clone)]
            pub struct [<$rust_name AttributeRef>]<'c, 't> {
                /// Handle that represents this [`Attribute`](crate::Attribute) in the MLIR C API.
                handle: ryft_xla_sys::bindings::MlirAttribute,

                /// [`Context`](crate::Context) that owns this [`Attribute`](crate::Attribute).
                context: &'c $crate::Context<'t>,
            }

            impl<'c, 't> [<$rust_name AttributeRef>]<'c, 't> {
                #[doc = "Returns the"]
                #[doc = $description]
                #[doc = "that is stored in this [`Attribute`](crate::Attribute)."]
                pub fn value(&self) -> $rust_name {
                    let value = unsafe {
                        $crate::StringRef::from_c_api(
                            ryft_xla_sys::bindings::[<$mlir_prefix $mlir_name AttrGetValue>](self.handle),
                        )
                    };
                    value
                        .as_str()
                        .ok()
                        .and_then(|value| $rust_name::try_from(value).ok())
                        .unwrap()
                }
            }

            $crate::mlir_subtype_trait_impls!(
                [<$rust_name AttributeRef>]<'c, 't> as Attribute,
                mlir_type = Attribute,
                mlir_subtype = [<$mlir_name Attr>],
                mlir_prefix = $mlir_prefix,
            );

            impl<'c, 't> $crate::FromWithContext<'c, 't, $rust_name> for [<$rust_name AttributeRef>]<'c, 't> {
                fn from_with_context(value: $rust_name, context: &'c $crate::Context<'t>) -> Self {
                    context.[<$($rust_prefix _)? $rust_name:snake>](value)
                }
            }

            impl<'t> $crate::Context<'t> {
                #[doc = "Creates a new"]
                #[doc = $description]
                #[doc = "owned by this [`Context`](crate::Context)."]
                pub fn [<$($rust_prefix _)? $rust_name:snake>]<'c>(
                    &'c self,
                    value: $rust_name,
                ) -> [<$rust_name AttributeRef>]<'c, 't> {
                    $(
                        // Make sure that the right dialect is loaded into this context to avoid segmentation faults.
                        self.load_dialect($crate::DialectHandle::$mlir_dialect_handle_constructor());
                    )?
                    // While this operation can mutate the context (in that it might add an entry to its corresponding
                    // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
                    // function quite inconvenient/annoying in practice. This should have no negative consequences in
                    // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
                    // should be no possibility for this function to cause problems with an immutable borrow.
                    let value = match value { $($rust_name::$rust_variant => $mlir_variant,)+ };
                    unsafe {
                        [<$rust_name AttributeRef>]::from_c_api(
                            ryft_xla_sys::bindings::[<$mlir_prefix $mlir_name AttrGet>](
                                *self.handle.borrow(),
                                $crate::StringRef::from(value).to_c_api(),
                            ),
                            &self,
                        )
                        .unwrap()
                    }
                }
            }
        }
    };
}

/// Generates accessor methods for fields of MLIR [`Attribute`](crate::Attribute) subtypes that call the
/// appropriate underlying MLIR C API functions to retrieve the corresponding attribute field values.
///
/// # Parameters
///
///   - `$rust_name`: Name of the Rust accessor method (e.g., `element_type`).
///   - `$mlir_name`: MLIR C API function name suffix (e.g., `ElementType`).
///   - `$ty`: Return type of the accessor method (e.g., `FloatTypeRef`, `[i64]`, or `i64`).
///   - `mlir_prefix`: Prefix for MLIR C API functions (e.g., `mlir`).
///
/// # Examples
///
/// ```ignore
/// impl<'c, 't> DotAlgorithmAttributeRef<'c, 't> {
///     // The following expands to:
///     //   pub fn lhs_precision_type(&self) -> FloatTypeRef<'c, 't> { ... }
///     mlir_attribute_field!(
///         lhs_precision_type,
///         DotAlgorithmGetLhsPrecisionType,
///         FloatTypeRef,
///         mlir_prefix = stablehlo,
///     );
///
///     // The following expands to:
///     //   pub fn lhs_component_count(&self) -> usize { ... }
///     mlir_attribute_field!(
///         lhs_component_count,
///         DotAlgorithmGetLhsComponentCount,
///         usize,
///         mlir_prefix = stablehlo,
///     );
/// }
/// ```
#[macro_export]
macro_rules! mlir_attribute_field {
    // We special-case `FloatTypeRef` because it requires a call to its constructor.
    ($rust_name:ident, $mlir_name:ident, FloatTypeRef, mlir_prefix = $mlir_prefix:ident $(,)*) => {
        paste::paste! {
            pub fn $rust_name(&self) -> $crate::FloatTypeRef<'c, 't> {
                unsafe {
                    $crate::FloatTypeRef::from_c_api(
                        ryft_xla_sys::bindings::[<$mlir_prefix $mlir_name>](self.handle),
                        self.context,
                    ).unwrap()
                }
            }
        }
    };
    // We also special-case array-valued fields because they require separately obtaining their size and allocating
    // a [`Vec`] before populating that vector with their values.
    ($rust_name:ident, $mlir_name:ident, [$ty:ty], mlir_prefix = $mlir_prefix:ident $(,)*) => {
        paste::paste! {
            pub fn $rust_name(&self) -> Vec<$ty> {
                unsafe {
                    let count = ryft_xla_sys::bindings::[<$mlir_prefix $mlir_name Size>](self.handle).cast_unsigned();
                    let mut values = Vec::with_capacity(count);
                    for i in 0..count {
                        let value = ryft_xla_sys::bindings::[<$mlir_prefix $mlir_name Elem>](
                            self.handle,
                            i.cast_signed(),
                        );
                        values.push(value as $ty);
                    }
                    values
                }
            }
        }
    };
    // All other field types are treated as simple accessors that forward the call to the C API and cast the result.
    ($rust_name:ident, $mlir_name:ident, $ty:ty, mlir_prefix = $mlir_prefix:ident $(,)*) => {
        paste::paste! {
            pub fn $rust_name(&self) -> $ty {
                unsafe { ryft_xla_sys::bindings::[<$mlir_prefix $mlir_name>](self.handle) as $ty }
            }
        }
    };
}

/// Generates the boilerplate code for MLIR [`Operation`](crate::Operation) subtypes.
///
/// This macro creates two operation subtypes:
///   - `Detached<Name>Operation`: Owned operation that manages its own lifetime and frees itself on drop.
///   - `<Name>OperationRef`: Borrowed reference to an operation owned by a [`Block`](crate::Block).
///
/// Both types implement the `<Name>Operation` trait (which must be defined separately by the user before invoking this
/// macro) and provide standard trait implementations for [`Operation`](crate::Operation), [`Clone`], [`PartialEq`],
/// [`Eq`], [`Hash`](std::hash::Hash), [`Display`](std::fmt::Display), and [`Debug`](std::fmt::Debug). It also provides
/// a [`Drop`] implementation for `Detached<Name>Operation` to properly clean up MLIR resources when it gets dropped.
///
/// # Requirements
///
/// Before using this macro, you must define a trait named `<Name>Operation<'o, 'c, 't>`. This trait should extend
/// [`Operation`](crate::Operation) and define methods for accessing operands, results, attributes, and any other
/// operation-specific properties.
///
/// # Parameters
///
///   - `$name`: The name of the operation (e.g., `Func`, `Return`, `Add`) that will be used to name the new
///     `Detached<Name>Operation` and `<Name>OperationRef` types.
///
/// # Example
///
/// ```ignore
/// // First, define a custom operation trait.
/// pub trait AddOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
///     /// Returns the left-hand side input of this [`AddOperation`].
///     fn lhs(&self) -> ValueRef<'o, 'c, 't> {
///         self.operand(0).unwrap()
///     }
///
///     /// Returns the right-hand side input of this [`AddOperation`].
///     fn rhs(&self) -> ValueRef<'o, 'c, 't> {
///         self.operand(1).unwrap()
///     }
/// }
///
/// // Then, invoke this macro to generate its concrete subtypes and their implementations.
/// // Specifically, this macro invocation will generate:
/// //   - `struct DetachedAddOperation<'c, 't: 'c> { ... }`,
/// //   - `struct AddOperationRef<'o, 'c: 'o, 't: 'c> { ... }`,
/// //   - `impl<'o, 'c: 'o, 't: 'c> AddOperation<'o, 'c, 't> for DetachedAddOperation<'c, 't> { ... }`,
/// //   - `impl<'o, 'c: 'o, 't: 'c> AddOperation<'o, 'c, 't> for AddOperationRef<'o, 'c, 't> { ... }`,
/// //   - `impl<'o, 'c: 'o, 't: 'c> Operation<'o, 'c, 't> for DetachedAddOperation<'c, 't> { ... }`,
/// //   - `impl<'o, 'c: 'o, 't: 'c> Operation<'o, 'c, 't> for AddOperationRef<'o, 'c, 't> { ... }`, and
/// //   - a bunch of other trait implementations for the new struct types.
/// mlir_op!(Add);
/// ```
#[macro_export]
macro_rules! mlir_op {
    ($name:ident) => {
        paste::paste! {
            pub struct [<Detached $name Operation>]<'c, 't: 'c> {
                /// Handle that represents this [`Operation`] in the MLIR C API.
                handle: ryft_xla_sys::bindings::MlirOperation,

                /// [`Context`] associated with this [`Operation`].
                context: &'c $crate::Context<'t>,
            }

            impl<'o, 'c: 'o, 't: 'c> [<$name Operation>]<'o, 'c, 't> for [<Detached $name Operation>]<'c, 't> {}

            impl<'o, 'c: 'o, 't: 'c> $crate::Operation<'o, 'c, 't> for [<Detached $name Operation>]<'c, 't> {
                unsafe fn from_c_api(
                    handle: ryft_xla_sys::bindings::MlirOperation,
                    context: &'c $crate::Context<'t>,
                ) -> Option<Self> {
                    if handle.ptr.is_null() { None } else { Some(Self { handle, context }) }
                }

                unsafe fn to_c_api(&self) -> ryft_xla_sys::bindings::MlirOperation {
                    self.handle
                }

                fn context(&self) -> &'c $crate::Context<'t> {
                    &self.context
                }
            }

            impl<'o, 'c: 'o, 't: 'c> $crate::DetachedOp<'o, 'c, 't> for [<Detached $name Operation>]<'c, 't> {}

            impl Clone for [<Detached $name Operation>]<'_, '_> {
                fn clone(&self) -> Self {
                    // The following context borrow ensures that access to the underlying MLIR data structures is done
                    // safely from Rust. It is maybe more conservative than would be ideal, but that is due to the
                    // limited exposure to MLIR internals that we have when working with the MLIR C API.
                    let _guard = self.context.borrow();
                    Self {
                        handle: unsafe { ryft_xla_sys::bindings::mlirOperationClone(self.handle) },
                        context: self.context,
                    }
                }
            }

            impl<'o, 'c: 'o, 't: 'c, O: $crate::Operation<'o, 'c, 't>> PartialEq<O>
                for [<Detached $name Operation>]<'c, 't>
            {
                fn eq(&self, other: &O) -> bool {
                    // The following context borrow ensures that access to the underlying MLIR data structures is done
                    // safely from Rust. It is maybe more conservative than would be ideal, but that is due to the
                    // limited exposure to MLIR internals that we have when working with the MLIR C API.
                    let _guard = self.context.borrow();
                    // Note that this function only checks for whether the two operation handles point to the same
                    // underlying operation. It does not perform a deep comparison of the contents of these operations.
                    unsafe { ryft_xla_sys::bindings::mlirOperationEqual(self.handle, other.to_c_api()) }
                }
            }

            impl Eq for [<Detached $name Operation>]<'_, '_> {}

            impl std::hash::Hash for [<Detached $name Operation>]<'_, '_> {
                fn hash<H: std::hash::Hasher>(&self, hasher: &mut H) {
                    unsafe { ryft_xla_sys::bindings::mlirOperationHashValue(self.handle).hash(hasher) }
                }
            }

            impl std::fmt::Display for [<Detached $name Operation>]<'_, '_> {
                fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                    // The following context borrow ensures that access to the underlying MLIR data structures is done
                    // safely from Rust. It is maybe more conservative than would be ideal, but that is due to the
                    // limited exposure to MLIR internals that we have when working with the MLIR C API.
                    let _guard = self.context.borrow();
                    let mut data = (formatter, Ok(()));
                    unsafe {
                        ryft_xla_sys::bindings::mlirOperationPrint(
                            <[<Detached $name Operation>] as $crate::Operation>::to_c_api(&self),
                            Some($crate::support::write_to_formatter_callback),
                            &mut data as *mut _ as *mut std::ffi::c_void,
                        );
                    }
                    data.1
                }
            }

            impl std::fmt::Debug for [<Detached $name Operation>]<'_, '_> {
                fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    write!(formatter, "{}[{}]", stringify!([<Detached $name Operation>]), self.to_string())
                }
            }

            impl Drop for [<Detached $name Operation>]<'_, '_> {
                fn drop(&mut self) {
                    if !self.handle.ptr.is_null() {
                        // The following context borrow ensures that access to the underlying MLIR data structures is
                        // done safely from Rust. It is maybe more conservative than would be ideal, but that is due to
                        // the limited exposure to MLIR internals that we have when working with the MLIR C API.
                        let _guard = self.context.borrow_mut();
                        unsafe { ryft_xla_sys::bindings::mlirOperationDestroy(self.handle) }
                    }
                }
            }

            #[derive(Copy, Clone)]
            pub struct [<$name OperationRef>]<'o, 'c: 'o, 't: 'c> {
                /// Handle that represents this [`Operation`] reference in the MLIR C API.
                handle: ryft_xla_sys::bindings::MlirOperation,

                /// [`Context`] associated with this [`Operation`] reference.
                context: &'c $crate::Context<'t>,

                /// [`PhantomData`] used to track the lifetime of the [`Block`] that owns the underlying [`Operation`].
                owner: std::marker::PhantomData<&'o ()>,
            }

            impl<'o, 'c: 'o, 't: 'c> [<$name Operation>]<'o, 'c, 't> for [<$name OperationRef>]<'o, 'c, 't> {}

            impl<'o, 'c: 'o, 't: 'c> $crate::Operation<'o, 'c, 't> for [<$name OperationRef>]<'o, 'c, 't> {
                unsafe fn from_c_api(
                    handle: ryft_xla_sys::bindings::MlirOperation,
                    context: &'c $crate::Context<'t>,
                ) -> Option<Self> {
                    if handle.ptr.is_null() {
                        None
                    } else {
                        Some(Self { handle, owner: std::marker::PhantomData, context })
                    }
                }

                unsafe fn to_c_api(&self) -> ryft_xla_sys::bindings::MlirOperation {
                    self.handle
                }

                fn context(&self) -> &'c $crate::Context<'t> {
                    &self.context
                }
            }

            impl<'o, 'c: 'o, 't: 'c> $crate::OpRef<'o, 'c, 't> for [<$name OperationRef>]<'o, 'c, 't> {}

            impl<'r, 'o, 'c: 'r, 't: 'c, O: $crate::Operation<'r, 'c, 't>> PartialEq<O>
                for [<$name OperationRef>]<'o, 'c, 't>
            {
                fn eq(&self, other: &O) -> bool {
                    // The following context borrow ensures that access to the underlying MLIR data structures is done
                    // safely from Rust. It is maybe more conservative than would be ideal, but that is due to the
                    // limited exposure to MLIR internals that we have when working with the MLIR C API.
                    let _guard = self.context.borrow();
                    // Note that this function only checks for whether the two operation handles point to the same
                    // underlying operation. It does not perform a deep comparison of the contents of these operations.
                    unsafe { ryft_xla_sys::bindings::mlirOperationEqual(self.handle, other.to_c_api()) }
                }
            }

            impl Eq for [<$name OperationRef>]<'_, '_, '_> {}

            impl std::hash::Hash for [<$name OperationRef>]<'_, '_, '_> {
                fn hash<H: std::hash::Hasher>(&self, hasher: &mut H) {
                    unsafe { ryft_xla_sys::bindings::mlirOperationHashValue(self.handle).hash(hasher) }
                }
            }

            impl std::fmt::Display for [<$name OperationRef>]<'_, '_, '_> {
                fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                    // The following context borrow ensures that access to the underlying MLIR data structures is done
                    // safely from Rust. It is maybe more conservative than would be ideal, but that is due to the
                    // limited exposure to MLIR internals that we have when working with the MLIR C API.
                    let _guard = self.context.borrow();
                    let mut data = (formatter, Ok(()));
                    unsafe {
                        ryft_xla_sys::bindings::mlirOperationPrint(
                            <[<$name OperationRef>] as $crate::Operation>::to_c_api(&self),
                            Some($crate::support::write_to_formatter_callback),
                            &mut data as *mut _ as *mut std::ffi::c_void,
                        );
                    }
                    data.1
                }
            }

            impl std::fmt::Debug for [<$name OperationRef>]<'_, '_, '_> {
                fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    write!(formatter, "{}[{}]", stringify!([<$name OperationRef>]), self.to_string())
                }
            }

            impl<'r, 'o: 'r, 'c: 'o, 't: 'c> From<&'r [<Detached $name Operation>]<'c, 't>>
                for [<$name OperationRef>]<'o, 'c, 't>
            {
                fn from(value: &'r [<Detached $name Operation>]<'c, 't>) -> Self {
                    unsafe {
                        <Self as $crate::Operation>::from_c_api(
                            <[<Detached $name Operation>] as $crate::Operation>::to_c_api(&value),
                            value.context,
                        ).unwrap()
                    }
                }
            }
        }
    };
}

/// Implements an operation trait for both detached and reference operation subtypes. This is a convenience macro that
/// implements a given trait for both the `Detached<OpName>Operation` and `<OpName>OperationRef` types that the
/// [`mlir_op!`](crate::mlir_op) macro generates. The provided trait can be either from the crate root
/// (e.g., [`OneResult`](crate::OneResult)) or a local trait defined (or imported) in the same module.
///
/// # Parameters
///
///   - `$op_name`: Name of the operation (e.g., `Add`, `Return`, etc.).
///   - `$trait_name`: Name of the trait to implement. If this is a local trait, then the name
///     must be prefixed with `@local`.
///
/// # Example
///
/// ```ignore
/// pub trait SineOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}
///
/// // The following macro invocations will generate struct types for the [`SineOperation`] subtypes along with
/// // trait implementations for those types, for all of the traits passed to the `mlir_op_trait!` macro.
/// mlir_op!(Sine);
/// mlir_op_trait!(Sine, OneOperand);
/// mlir_op_trait!(Sine, OneResult);
/// mlir_op_trait!(Sine, ZeroRegions);
/// mlir_op_trait!(Sine, ZeroSuccessors);
/// mlir_op_trait!(Sine, @local HasAccuracy);
/// ```
#[macro_export]
macro_rules! mlir_op_trait {
    ($op_name:ident, $trait_name:ident) => {
        paste::paste! {
            impl<'o, 'c: 'o, 't: 'c> $crate::$trait_name<'o, 'c, 't> for [<Detached $op_name Operation>]<'c, 't> {}
            impl<'o, 'c: 'o, 't: 'c> $crate::$trait_name<'o, 'c, 't> for [<$op_name OperationRef>]<'o, 'c, 't> {}
        }
    };
    ($op_name:ident, @local $trait_name:ident) => {
        paste::paste! {
            impl<'o, 'c: 'o, 't: 'c> $trait_name<'o, 'c, 't> for [<Detached $op_name Operation>]<'c, 't> {}
            impl<'o, 'c: 'o, 't: 'c> $trait_name<'o, 'c, 't> for [<$op_name OperationRef>]<'o, 'c, 't> {}
        }
    };
}

/// Generates the structs, traits, and constructor necessary for representing a unary MLIR operation with a single
/// output which requires explicit specification of its output type.
///
/// This macro generates:
///   - a trait for representing the new operation named `<Op>Operation`,
///   - an invocation to the [`mlir_op!`](crate::mlir_op) macro to generate the necessary boilerplate, and
///   - a constructor function for instances of this new operation type.
///
/// Use this macro for unary operations where the output type cannot be inferred from the input type alone
/// (e.g., type conversions, casts, or operations that change the shape/type of the input).
///
/// # Parameters
///
///   - `$dialect`: MLIR dialect name (e.g., `arith`, `func`, etc.).
///   - `$op`: Operation name (e.g., `trunci`, `sitofp`, etc.).
///
/// # Example
///
/// ```ignore
/// // This macro invocation will generate:
/// //   - `trait BitcastOperation<'o, 'c, 't>`,
/// //   - an invocation to `mlir_op!(Bitcast)`,
/// //   - an invocation to `mlir_op_trait!(Bitcast, OneOperand)`,
/// //   - an invocation to `mlir_op_trait!(Bitcast, OneResult)`, and
/// //   - a constructor function called `bitcast` that takes an input [`Value`], an output [`Type`], and a [`Location`]
/// //     and returns a new [`DetachedBitcastOperation`].
/// mlir_generic_unary_op!(arith, bitcast);
/// ```
#[macro_export]
macro_rules! mlir_generic_unary_op {
    ($dialect:ident, $op:ident) => {
        paste::paste! {
            #[doc = "Trait representing the `" $dialect:lower "." $op "` operation."]
            pub trait [<$op:camel Operation>]<'o, 'c: 'o, 't: 'c>: $crate::Operation<'o, 'c, 't> {}

            $crate::mlir_op!([<$op:camel>]);
            $crate::mlir_op_trait!([<$op:camel>], OneOperand);
            $crate::mlir_op_trait!([<$op:camel>], OneResult);

            #[doc = "Constructs a new detached/owned [`" [<$op:camel Operation>] "`] at the specified [`Location`]."]
            #[doc = "Note that if any of the inputs to this function are invalid, it will panic!"]
            pub fn $op<
                'v,
                'c: 'v,
                't: 'c,
                V: $crate::Value<'v, 'c, 't>,
                T: $crate::Type<'c, 't>,
                L: $crate::Location<'c, 't>,
            >(
                input: V,
                output_type: T,
                location: L,
            ) -> [<Detached $op:camel Operation>]<'c, 't> {
                let context = location.context();
                context.load_dialect($crate::DialectHandle::$dialect());
                let name = format!("{}.{}", stringify!($dialect), stringify!($op));
                $crate::OperationBuilder::new(name.as_str(), location)
                    .add_operands(&[input])
                    .add_result(output_type)
                    .build()
                    .and_then(|operation| unsafe { $crate::DetachedOp::cast(operation) })
                    .expect(format!("invalid arguments to `{}::{}`", stringify!($dialect), stringify!($op)).as_str())
            }
        }
    };
}

/// Generates the structs, traits, and constructor necessary for representing a unary MLIR operation with a single
/// output which does not require explicit specification of its output type (i.e., it relies on type inference).
///
/// This macro generates:
///   - a trait for representing the new operation named `<Op>Operation`,
///   - an invocation to the [`mlir_op!`](crate::mlir_op) macro to generate the necessary boilerplate, and
///   - a constructor function for instances of this new operation type.
///
/// Use this macro for unary operations where the output type can be inferred from the input type alone.
///
/// # Parameters
///
///   - `$dialect`: MLIR dialect name (e.g., `arith`, `func`, etc.).
///   - `$op`: Operation name (e.g., `negf`, `sign`, etc.).
///
/// # Example
///
/// ```ignore
/// // This macro invocation will generate:
/// //   - `trait NegfOperation<'o, 'c, 't>`,
/// //   - an invocation to `mlir_op!(Negf)`,
/// //   - an invocation to `mlir_op_trait!(Negf, OneOperand)`,
/// //   - an invocation to `mlir_op_trait!(Negf, OneResult)`, and
/// //   - a constructor function called `negf` that takes an input [`Value`] and a [`Location`]
/// //     and returns a new [`DetachedNegfOperation`].
/// mlir_unary_op!(arith, negf);
/// ```
#[macro_export]
macro_rules! mlir_unary_op {
    ($dialect:ident, $op:ident) => {
        paste::paste! {
            #[doc = "Trait representing the `" $dialect:lower "." $op "` operation."]
            pub trait [<$op:camel Operation>]<'o, 'c: 'o, 't: 'c>: $crate::Operation<'o, 'c, 't> {}

            $crate::mlir_op!([<$op:camel>]);
            $crate::mlir_op_trait!([<$op:camel>], OneOperand);
            $crate::mlir_op_trait!([<$op:camel>], OneResult);

            #[doc = "Constructs a new detached/owned [`" [<$op:camel Operation>] "`] at the specified [`Location`]."]
            #[doc = "Note that if any of the inputs to this function are invalid, it will panic!"]
            pub fn $op<'v, 'c: 'v, 't: 'c, V: $crate::Value<'v, 'c, 't>, L: $crate::Location<'c, 't>>(
                input: V,
                location: L,
            ) -> [<Detached $op:camel Operation>]<'c, 'c> {
                let context = location.context();
                context.load_dialect($crate::DialectHandle::$dialect());
                let name = format!("{}.{}", stringify!($dialect), stringify!($op));
                $crate::OperationBuilder::new(name.as_str(), location)
                    .add_operands(&[input])
                    .enable_result_type_inference()
                    .build()
                    .and_then(|operation| unsafe { $crate::DetachedOp::cast(operation) })
                    .expect(format!("invalid arguments to `{}::{}`", stringify!($dialect), stringify!($op)).as_str())
            }
        }
    };
}

/// Generates the structs, traits, and constructor necessary for representing a binary MLIR operation with a single
/// output which does not require explicit specification of its output type (i.e., it relies on type inference).
///
/// This macro generates:
///   - a trait for representing the new operation named `<Op>Operation`,
///   - an invocation to the [`mlir_op!`](crate::mlir_op) macro to generate the necessary boilerplate, and
///   - a constructor function for instances of this new operation type.
///
/// Use this macro for binary operations where the output type can be inferred from the input type alone.
///
/// # Parameters
///
///   - `$dialect`: MLIR dialect name (e.g., `arith`, `func`, etc.).
///   - `$op`: Operation name (e.g., `add`, `sub`, etc.).
///
/// # Example
///
/// ```ignore
/// // This macro invocation will generate:
/// //   - `trait AddfOperation<'o, 'c, 't>`,
/// //   - an invocation to `mlir_op!(Addf)`,
/// //   - an invocation to `mlir_op_trait!(Addf, OneOperand)`,
/// //   - an invocation to `mlir_op_trait!(Addf, OneResult)`, and
/// //   - a constructor function called `addf` that takes two input [`Value`]s and a [`Location`]
/// //     and returns a new [`DetachedAddfOperation`].
/// mlir_binary_op!(arith, addf);
/// ```
#[macro_export]
macro_rules! mlir_binary_op {
    ($dialect:ident, $op:ident) => {
        paste::paste! {
            #[doc = "Operation trait for the `" $dialect:lower "." $op "` operation."]
            pub trait [<$op:camel Operation>]<'o, 'c: 'o, 't: 'c>: $crate::Operation<'o, 'c, 't> {
                #[doc = "Returns the left-hand side input (i.e., first operand)"]
                #[doc = "of this [`" [<$op:camel Operation>] "`]."]
                fn lhs(&self) -> $crate::ValueRef<'o, 'c, 't> {
                    self.operand(0).unwrap()
                }

                #[doc = "Returns the right-hand side input (i.e., second operand)"]
                #[doc = "of this [`" [<$op:camel Operation>] "`]."]
                fn rhs(&self) -> $crate::ValueRef<'o, 'c, 't> {
                    self.operand(1).unwrap()
                }
            }

            $crate::mlir_op!([<$op:camel>]);
            $crate::mlir_op_trait!([<$op:camel>], OneResult);

            #[doc = "Constructs a new detached/owned [`" [<$op:camel Operation>] "`] at the specified [`Location`]."]
            #[doc = "Note that if any of the inputs to this function are invalid, it will panic!"]
            pub fn $op<
                'lhs,
                'rhs,
                'c: 'lhs + 'rhs,
                't: 'c,
                LHS: $crate::Value<'lhs, 'c, 't>,
                RHS: $crate::Value<'rhs, 'c, 't>,
                L: $crate::Location<'c, 't>,
            >(
                lhs: LHS,
                rhs: RHS,
                location: L,
            ) -> [<Detached $op:camel Operation>]<'c, 't> {
                let context = location.context();
                context.load_dialect($crate::DialectHandle::$dialect());
                let name = format!("{}.{}", stringify!($dialect), stringify!($op));
                $crate::OperationBuilder::new(name.as_str(), location)
                    .add_operands(&[lhs.as_ref(), rhs.as_ref()])
                    .enable_result_type_inference()
                    .build()
                    .and_then(|operation| unsafe { $crate::DetachedOp::cast(operation) })
                    .expect(format!("invalid arguments to `{}::{}`", stringify!($dialect), stringify!($op)).as_str())
            }
        }
    };
}

/// Generates constructor and registration functions for MLIR compiler [`Pass`](crate::Pass)es.
///
/// This macro generates two functions for working with MLIR passes:
///   - a `create_<name>` function that creates a new [`Pass`](crate::Pass) instance, and
///   - a `register_<name>` function that registers the pass with the global pass registry (idempotent and thread-safe).
///
/// The registration function uses [`OnceLock`](std::sync::OnceLock) to ensure the pass is registered at most once,
/// even when called from multiple threads. It also uses [`GLOBAL_REGISTRATION_MUTEX`](crate::GLOBAL_REGISTRATION_MUTEX)
/// to ensure thread-safety when interacting with MLIR's global state.
///
/// # Parameters
///
///   - `$rust_name`: Rust function name suffix (e.g., `canonicalizer` for `create_canonicalizer`).
///   - `$mlir_name`: MLIR C API function name suffix (e.g., `Canonicalizer` for `mlirCreateCanonicalizer`).
///
/// # Example
///
/// ```ignore
/// // This macro invocation will generate:
/// //  - a `create_conversion_arith_to_llvm_pass` function that invokes the MLIR C API
/// //    `mlirCreateConversionArithToLLVMConversionPass` function under the hood, and
/// //  - a `register_conversion_arith_to_llvm_pass` function that invokes the MLIR C API
/// //    `mlirRegisterConversionArithToLLVMConversionPass` function under the hood.
/// mlir_pass!(conversion_arith_to_llvm_pass, ConversionArithToLLVMConversionPass);
/// ```
#[macro_export]
macro_rules! mlir_pass {
    ($rust_name:ident, $mlir_name:ident) => {
        paste::paste! {
            pub fn [<create_ $rust_name>]() -> $crate::Pass {
                unsafe { $crate::Pass::from_c_api(ryft_xla_sys::bindings::[<mlirCreate $mlir_name>]()).unwrap() }
            }

            pub fn [<register_ $rust_name>]() {
                // Use [`OnceLock`] to ensure that the pass registration function is called at most once.
                static INITIALIZED: OnceLock<()> = OnceLock::new();
                INITIALIZED.get_or_init(|| unsafe {
                    let _guard = $crate::GLOBAL_REGISTRATION_MUTEX.lock();
                    ryft_xla_sys::bindings::[<mlirRegister $mlir_name>]()
                });
            }
        }
    };
}
