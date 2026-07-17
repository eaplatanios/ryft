use crate::programs::atoms::AtomId;
use crate::programs::regions::RegionId;

/// Location of one [`Instruction`] in a multi-region [`Program`](crate::Program), identified by its containing
/// [`Region`](crate::Region) and its zero-based index within that region's instruction sequence.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InstructionId {
    /// [`Region`](crate::Region) containing the instruction.
    region: RegionId,

    /// Zero-based instruction index within the containing [`Region`](crate::Region).
    index: usize,
}

impl InstructionId {
    /// Creates a new [`InstructionId`] from the provided containing region and instruction index.
    #[inline]
    pub fn new(region: RegionId, index: usize) -> Self {
        Self { region, index }
    }

    /// Returns the [`RegionId`] of the [`Region`](crate::Region) containing the instruction.
    #[inline]
    pub fn region(self) -> RegionId {
        self.region
    }

    /// Returns the zero-based instruction index within the containing [`Region`](crate::Region).
    #[inline]
    pub fn index(self) -> usize {
        self.index
    }
}
/// [`Instruction`]s represent applications of [`Operation`](crate::Operation)s to input values in
/// [`Program`](crate::Program)s. Each [`Region`](crate::Region) executes its [`Instruction`]s in sequential order.
/// Beyond its operation and its input and output [`Atom`](crate::Atom)s, an instruction carries the [`RegionId`]s of
/// the nested computations attached to the application (e.g., the `true`/`false` branches of a condition, a scan body,
/// or the shared program of a JIT call), in the operation-defined order. Note that there is one
/// [`Region`](crate::Region) edge kind, and sharing is expressed directly in the graph. Several [`Instruction`]s may
/// reference the same [`RegionId`], and a region stays alive for as long as it is reachable from the entry region. What
/// a slot *means* (i.e., a branch-like computation that lowers inline versus a call-like computation that lowers and
/// compiles once as a shared function) is defined by the operation and not by the edge. For example,
/// `if p { f(x) + f(2 * x) } else { x }` with a JIT-compiled `f` is one condition instruction attaching a `true` and a
/// `false` branch [`Region`](crate::Region), where the `true` branch contains two call instructions that both reference
/// the single region holding `f`'s body  (i.e., one shared region, three region edges, and the inline-versus-shared
/// lowering decision carried by the condition and call operations, respectively). Two structurally equal but
/// independently created computations remain distinct regions, because [`ProgramBuilder`](crate::ProgramBuilder)
/// imports regions by *identity* (i.e., [`import_region`](crate::ProgramBuilder::import_region) always copies and
/// [`intern_callee`](crate::ProgramBuilder::intern_callee) interns by [`Rc`](std::rc::Rc) identity),
/// never by structure.
#[derive(Clone, Debug)]
pub struct Instruction<O> {
    /// [`Operation`](crate::Operation) applied by this [`Instruction`].
    pub(crate) operation: O,

    /// [`AtomId`]s of the input [`Atom`](crate::Atom)s consumed by this [`Instruction`].
    pub(crate) inputs: Vec<AtomId>,

    /// [`AtomId`]s of the output [`Atom`](crate::Atom)s produced by this [`Instruction`].
    pub(crate) outputs: Vec<AtomId>,

    /// [`RegionId`]s of the nested computations attached to this [`Instruction`], in the operation-defined order.
    pub(crate) regions: Vec<RegionId>,
}

impl<O> Instruction<O> {
    /// Creates a new [`Instruction`].
    #[inline]
    pub fn new(operation: O, inputs: Vec<AtomId>, outputs: Vec<AtomId>, regions: Vec<RegionId>) -> Self {
        Self { operation, inputs, outputs, regions }
    }

    /// Returns the [`Operation`](crate::Operation) applied by this [`Instruction`].
    #[inline]
    pub fn operation(&self) -> &O {
        &self.operation
    }

    /// Returns the [`AtomId`]s of the input [`Atom`](crate::Atom)s consumed by this [`Instruction`].
    #[inline]
    pub fn inputs(&self) -> &[AtomId] {
        self.inputs.as_slice()
    }

    /// Returns the [`AtomId`]s of the output [`Atom`](crate::Atom)s produced by this [`Instruction`].
    #[inline]
    pub fn outputs(&self) -> &[AtomId] {
        self.outputs.as_slice()
    }

    /// Returns the [`RegionId`]s of the nested computations attached to this [`Instruction`],
    /// in the operation-defined order.
    #[inline]
    pub fn regions(&self) -> &[RegionId] {
        &self.regions
    }

    /// Consumes this [`Instruction`] and returns its [`Operation`](crate::Operation), input [`AtomId`]s,
    /// output [`AtomId`]s, and attached region [`RegionId`]s.
    #[inline]
    pub fn into_parts(self) -> (O, Vec<AtomId>, Vec<AtomId>, Vec<RegionId>) {
        (self.operation, self.inputs, self.outputs, self.regions)
    }
}
