use ryft_xla_sys::bindings::{
    MlirLocation, mlirLocationFileLineColGet, mlirLocationFileLineColRangeGet,
    mlirLocationFileLineColRangeGetEndColumn, mlirLocationFileLineColRangeGetEndLine,
    mlirLocationFileLineColRangeGetFilename, mlirLocationFileLineColRangeGetStartColumn,
    mlirLocationFileLineColRangeGetStartLine, mlirLocationFileLineColRangeGetTypeID,
};

use crate::{Context, Identifier, Location, StringRef, TypeId, mlir_subtype_trait_impls};

/// [`Location`] at specific lines and columns within a file. These properties are available via functions in
/// this struct (e.g., [`FileLocationRef::file_name`], [`FileLocationRef::start_line`],
/// [`FileLocationRef::start_column`], [`FileLocationRef::end_line`], and [`FileLocationRef::end_column`]).
/// Refer to the [MLIR documentation](https://mlir.llvm.org/docs/Dialects/Builtin/#filelinecolrange)
/// for more information.
#[derive(Copy, Clone)]
pub struct FileLocationRef<'c, 't> {
    /// Handle that represents this [`Location`] in the MLIR C API.
    handle: MlirLocation,

    /// [`Context`] that owns this [`Location`].
    context: &'c Context<'t>,
}

impl<'c, 't> FileLocationRef<'c, 't> {
    /// Returns the [`TypeId`] that corresponds to [`FileLocationRef`].
    pub fn type_id() -> TypeId<'static> {
        unsafe { TypeId::from_c_api(mlirLocationFileLineColRangeGetTypeID()).unwrap() }
    }

    /// Returns the file name of this [`FileLocationRef`].
    pub fn file_name(&self) -> Identifier<'c, 't> {
        unsafe { Identifier::from_c_api(mlirLocationFileLineColRangeGetFilename(self.handle)) }
    }

    /// Returns the start line of this [`FileLocationRef`].
    pub fn start_line(&self) -> usize {
        unsafe { mlirLocationFileLineColRangeGetStartLine(self.handle) as usize }
    }

    /// Returns the start column of this [`FileLocationRef`].
    pub fn start_column(&self) -> usize {
        unsafe { mlirLocationFileLineColRangeGetStartColumn(self.handle) as usize }
    }

    /// Returns the end line of this [`FileLocationRef`].
    pub fn end_line(&self) -> usize {
        unsafe { mlirLocationFileLineColRangeGetEndLine(self.handle) as usize }
    }

    /// Returns the end column of this [`Location`].
    pub fn end_column(&self) -> usize {
        unsafe { mlirLocationFileLineColRangeGetEndColumn(self.handle) as usize }
    }
}

mlir_subtype_trait_impls!(FileLocationRef<'c, 't> as Location, mlir_type = Location, mlir_subtype = FileLineColRange);

impl<'t> Context<'t> {
    /// Creates a new [`FileLocationRef`] for the specified line and column at the provided file,
    /// owned by this [`Context`].
    pub fn file_location<'c, S: AsRef<str>>(
        &'c self,
        filename: S,
        line: usize,
        column: usize,
    ) -> FileLocationRef<'c, 't> {
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        unsafe {
            FileLocationRef::from_c_api(
                mlirLocationFileLineColGet(
                    *self.handle.borrow(),
                    StringRef::from(filename.as_ref()).to_c_api(),
                    line as u32,
                    column as u32,
                ),
                self,
            )
            .unwrap()
        }
    }

    /// Creates a new [`FileLocationRef`] for the specified line and column range at the provided file,
    /// owned by this [`Context`].
    pub fn file_range_location<'c, S: AsRef<str>>(
        &'c self,
        filename: S,
        start_line: usize,
        start_column: usize,
        end_line: usize,
        end_column: usize,
    ) -> FileLocationRef<'c, 't> {
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        unsafe {
            FileLocationRef::from_c_api(
                mlirLocationFileLineColRangeGet(
                    *self.handle.borrow(),
                    StringRef::from(filename.as_ref()).to_c_api(),
                    start_line as u32,
                    start_column as u32,
                    end_line as u32,
                    end_column as u32,
                ),
                self,
            )
            .unwrap()
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::locations::tests::{test_location_casting, test_location_display_and_debug};

    use super::*;

    #[test]
    fn test_file_location_type_id() {
        let file_location_type_id = FileLocationRef::type_id();
        assert_eq!(FileLocationRef::type_id(), FileLocationRef::type_id());
        assert_eq!(file_location_type_id, FileLocationRef::type_id());
    }

    #[test]
    fn test_file_location() {
        let context = Context::new();
        let location = context.file_range_location("main.rs", 5, 10, 7, 20);
        assert_eq!(&context, location.context());
        assert_eq!(location.file_name().as_str().unwrap(), "main.rs");
        assert_eq!(location.start_line(), 5);
        assert_eq!(location.start_column(), 10);
        assert_eq!(location.end_line(), 7);
        assert_eq!(location.end_column(), 20);
    }

    #[test]
    fn test_file_location_equality() {
        let context = Context::new();

        // Same locations from the same context must be equal because they are "uniqued".
        let location_1 = context.file_location("test.rs", 10, 5);
        let location_2 = context.file_location("test.rs", 10, 5);
        assert_eq!(location_1, location_2);

        // Different locations from the same context must not be equal.
        let location_2 = context.file_location("test.rs", 11, 5);
        assert_ne!(location_1, location_2);

        // Same locations from different contexts must not be equal.
        let context = Context::new();
        let location_2 = context.file_location("test.rs", 10, 5);
        assert_ne!(location_1, location_2);
    }

    #[test]
    fn test_file_location_display_and_debug() {
        let context = Context::new();
        let location = context.file_location("test.rs", 42, 10);
        test_location_display_and_debug(location, "loc(\"test.rs\":42:10)");
    }

    #[test]
    fn test_file_range_location_display_and_debug() {
        let context = Context::new();
        let location = context.file_range_location("main.rs", 5, 10, 7, 20);
        test_location_display_and_debug(location, "loc(\"main.rs\":5:10 to 7:20)");
    }

    #[test]
    fn test_file_location_casting() {
        let context = Context::new();
        let location = context.file_location("test.rs", 42, 10);
        test_location_casting(location);
    }
}
