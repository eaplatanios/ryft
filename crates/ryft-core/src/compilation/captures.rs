use std::borrow::Cow;
use std::fmt::{Debug, Display};

use ryft_macros::Parameter;

use crate::operations::Operation;
use crate::parameters::{Parameter, Parameterized};
use crate::tracing::{Program, Traceable, Value};
use crate::types::{Type, Typed};

/// Reference to a value captured outside a staged [`Program`].
///
/// The program stores only this lifetime-free reference in its atom table. The corresponding
/// runtime value lives in the surrounding [`CapturedProgram`] capture table at [`Self::index`].
/// The IR remains abstract and reusable, while concrete runtime values stay in a side
/// environment owned by the compiled function.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub struct CapturedConstant<T: Type + Parameter> {
    /// Index into the surrounding capture table.
    index: usize,

    /// Abstract type metadata for the captured value.
    r#type: T,
}

impl<T: Type + Parameter> CapturedConstant<T> {
    /// Creates a captured-constant reference.
    #[inline]
    pub fn new(index: usize, r#type: T) -> Self {
        Self { index, r#type }
    }

    /// Returns the index into the surrounding capture table.
    #[inline]
    pub fn index(&self) -> usize {
        self.index
    }
}

impl<T: Type + Parameter> Display for CapturedConstant<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "capture#{}:{}", self.index, self.r#type)
    }
}

impl<T: Type + Parameter> Typed<T> for CapturedConstant<T> {
    #[inline]
    fn r#type(&self) -> Cow<'_, T> {
        Cow::Borrowed(&self.r#type)
    }
}

impl<T: Type + Parameter> Traceable<T> for CapturedConstant<T> {}

/// Captured constants are value carriers for staged programs: they identify runtime values
/// stored outside the IR rather than containing those values directly.
impl<T: Type + Parameter> Value<T> for CapturedConstant<T> {}

/// A staged [`Program`] paired with the concrete runtime values referenced by its captured
/// constants.
///
/// `Program` remains lifetime-free except for its operation payloads. Concrete values of type
/// `V` live only in [`Self::captures`], and atom-table constants are
/// [`CapturedConstant<T>`] references into that side table.
#[derive(Clone)]
pub struct CapturedProgram<
    T: Type + Parameter,
    V: Traceable<T>,
    O,
    Input: Parameterized<CapturedConstant<T>>,
    Output: Parameterized<CapturedConstant<T>>,
> {
    /// Staged program whose constants are capture references.
    program: Program<T, CapturedConstant<T>, O, Input, Output>,

    /// Concrete captured values referenced by [`CapturedConstant`] indices in [`Self::program`].
    captures: Vec<V>,
}

impl<
    T: Type + Parameter,
    V: Traceable<T>,
    O,
    Input: Parameterized<CapturedConstant<T>>,
    Output: Parameterized<CapturedConstant<T>>,
> Debug for CapturedProgram<T, V, O, Input, Output>
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CapturedProgram")
            .field("captures", &self.captures.len())
            .finish_non_exhaustive()
    }
}

impl<
    T: Type + Parameter,
    V: Traceable<T>,
    O,
    Input: Parameterized<CapturedConstant<T>>,
    Output: Parameterized<CapturedConstant<T>>,
> CapturedProgram<T, V, O, Input, Output>
{
    /// Creates a captured program from an already capture-referenced program and capture table.
    #[inline]
    pub fn new(program: Program<T, CapturedConstant<T>, O, Input, Output>, captures: Vec<V>) -> Self {
        Self { program, captures }
    }

    /// Returns the staged program.
    #[inline]
    pub fn program(&self) -> &Program<T, CapturedConstant<T>, O, Input, Output> {
        &self.program
    }

    /// Returns the captured runtime values.
    #[inline]
    pub fn captures(&self) -> &[V] {
        self.captures.as_slice()
    }

    /// Returns the abstract types of the captured runtime values.
    #[inline]
    pub fn capture_types(&self) -> Vec<T> {
        self.captures.iter().map(|capture| capture.r#type().into_owned()).collect()
    }

    /// Consumes this wrapper and returns the staged program and capture table.
    #[inline]
    pub fn into_parts(self) -> (Program<T, CapturedConstant<T>, O, Input, Output>, Vec<V>) {
        (self.program, self.captures)
    }
}

impl<
    T: Type + Parameter,
    V: Traceable<T>,
    O: Clone + Operation<T>,
    Input: Parameterized<CapturedConstant<T>>,
    Output: Parameterized<CapturedConstant<T>>,
> CapturedProgram<T, V, O, Input, Output>
{
    /// Returns a cloned flat view of the wrapped program while preserving the capture table.
    #[inline]
    pub fn to_flat_program(&self) -> CapturedProgram<T, V, O, Vec<CapturedConstant<T>>, Vec<CapturedConstant<T>>> {
        CapturedProgram::new(self.program.to_flat_program(), self.captures.clone())
    }
}

impl<
    T: Type + Parameter,
    V: Traceable<T>,
    O: Operation<T>,
    Input: Parameterized<CapturedConstant<T>>,
    Output: Parameterized<CapturedConstant<T>>,
> CapturedProgram<T, V, O, Input, Output>
{
    /// Consumes this wrapper and returns a flat wrapped program without cloning the IR or captures.
    #[inline]
    pub fn into_flat_program(self) -> CapturedProgram<T, V, O, Vec<CapturedConstant<T>>, Vec<CapturedConstant<T>>> {
        CapturedProgram::new(self.program.into_flat_program(), self.captures)
    }
}
