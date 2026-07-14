//! Phase 0 compile prototype for the first-class-program-regions plan (`.tasks/plan_first_class_program_regions.md`).
//!
//! This file is a self-contained model of the FUTURE machinery shape. It exists to prove, before any production
//! migration, that the post-migration design is solver-sound and borrow-sound:
//!
//!   1. An operation enum whose higher-order variant carries NO nested program (regions are instruction
//!      attachments) compiles with ordinary direct per-variant bounds and no recursive-variant where-clause filtering
//!      for interpretation, partial evaluation, batching, fused JVP,
//!      split linearization, and transposition, including the fresh-context driver stacks that diverge today
//!      (`BatchingContext<TracingContext<..>>`, `DifferentiationContext<PartialEvaluationContext<TracingContext<..>>>`).
//!   2. Region access is a separate, mandatory, call-scoped driver argument. Rules never see a recursive bound; the
//!      driver constructs the nested-work callback where the family bound is already established. Interpretation's
//!      context parameter `C` stays deliberately unbounded.
//!   3. Immutable instruction/region views coexist with recursive driver work and with a `&mut` destination
//!      context (transposition) without cloning programs.
//!   4. The region-carrying `Context::bind` protocol covers freshly authored regional
//!      operations in eager and staging contexts, delegating to `bind` when attachment-free.
//!   5. Builder imports copy the complete reachable closure with sharing preserved, and callee imports intern by
//!      live `Rc` identity.
//!   6. Lazy residual-origin resolution (`OutputRegionProvenance`) recovers producers through region-attached
//!      instructions with first-occurrence deduplication, empty origins for inputs/constants, and errors for
//!      malformed rules.
//!
//! Semantics here are deliberately minimal (batching/JVP/transposition bodies are stubs); the *generic structure* —
//! trait shapes, bounds, context stacks, borrow shapes — mirrors the production machinery, because that structure is
//! what the Phase 0 gate is about. Stored constants (`Stored`) differ from runtime values and include a capture
//! reference so constant-to-runtime substitution cannot be accidentally erased by the model.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

/// Everything inside `model` plays the role of `ryft-core` internals; the `#[test]` functions at the bottom play
/// the role of downstream user code. Concrete drivers remain module-private, while operation rules depend only on
/// the public driver contracts.
mod model {
    use super::*;

    pub type R<T> = Result<T, String>;

    // ------------------------------------------------------------------------------------------------------------
    // IR: program, regions, instructions, locators.
    // ------------------------------------------------------------------------------------------------------------

    /// Region index scoped to one immutable [`Program`].
    #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
    pub struct RegionId(pub usize);

    /// Location of one SSA value in a multi-region program.
    #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
    pub struct ValueId {
        pub region: RegionId,
        pub atom: usize,
    }

    /// Stored-constant type, deliberately distinct from every runtime value type, with a capture reference so the
    /// prototype exercises constant substitution through nested regions.
    #[derive(Clone, Debug, PartialEq)]
    pub enum Stored {
        Literal(f64),
        Capture(usize),
    }

    #[derive(Clone, Debug)]
    pub enum Atom<C> {
        Constant(C),
        Variable,
    }

    #[derive(Clone, Debug)]
    pub struct Instruction<O> {
        pub operation: O,
        pub inputs: Vec<usize>,
        pub outputs: Vec<usize>,
        pub regions: Vec<RegionId>,
    }

    /// One sealed flat computation region. Only [`ProgramBuilder`] creates these (via a consumed
    /// [`RegionBuilder`]), so an attached region can never change after an instruction referencing it was built.
    #[derive(Clone, Debug)]
    pub struct Region<C, O> {
        pub atoms: Vec<Atom<C>>,
        pub instructions: Vec<Instruction<O>>,
        pub inputs: Vec<usize>,
        pub outputs: Vec<usize>,
    }

    #[derive(Clone, Debug)]
    pub struct Program<C, O> {
        pub regions: Vec<Region<C, O>>,
        pub entry: RegionId,
    }

    /// Borrowed view of one region in a [`Program`].
    pub struct RegionRef<'a, C, O> {
        /// Program that owns the region arena.
        program: &'a Program<C, O>,

        /// Identifier of the viewed region in `program`.
        id: RegionId,
    }

    impl<C: Clone, O: Clone> Program<C, O> {
        pub fn entry_region(&self) -> &Region<C, O> {
            &self.regions[self.entry.0]
        }

        /// Returns a borrowed view of `region`.
        pub fn region_ref(&self, region: RegionId) -> RegionRef<'_, C, O> {
            RegionRef { program: self, id: region }
        }
    }

    impl<C: Clone, O: Clone> RegionRef<'_, C, O> {
        /// Materializes this region and its reachable closure as a standalone program.
        pub fn into_program(self) -> Program<C, O> {
            let mut builder = ProgramBuilder::new();
            let entry = builder.add_region_closure(self.program, self.id, &mut HashMap::new());
            builder.build(entry)
        }
    }

    // ------------------------------------------------------------------------------------------------------------
    // Builder: sealed regions, full-closure imports, Rc-interned callees.
    // ------------------------------------------------------------------------------------------------------------

    /// Builds one region; consumed by [`ProgramBuilder::seal`] so attached regions are immutable by construction.
    pub struct RegionBuilder<C, O> {
        region: Region<C, O>,
    }

    impl<C: Clone, O: Clone> RegionBuilder<C, O> {
        pub fn new() -> Self {
            Self {
                region: Region { atoms: Vec::new(), instructions: Vec::new(), inputs: Vec::new(), outputs: Vec::new() },
            }
        }

        pub fn add_input(&mut self) -> usize {
            let id = self.region.atoms.len();
            self.region.atoms.push(Atom::Variable);
            self.region.inputs.push(id);
            id
        }

        pub fn add_constant(&mut self, constant: C) -> usize {
            let id = self.region.atoms.len();
            self.region.atoms.push(Atom::Constant(constant));
            id
        }

        pub fn add_instruction(
            &mut self,
            operation: O,
            inputs: Vec<usize>,
            output_count: usize,
            regions: Vec<RegionId>,
        ) -> Vec<usize> {
            let outputs = (0..output_count)
                .map(|_| {
                    let id = self.region.atoms.len();
                    self.region.atoms.push(Atom::Variable);
                    id
                })
                .collect::<Vec<_>>();
            self.region.instructions.push(Instruction { operation, inputs, outputs: outputs.clone(), regions });
            outputs
        }

        pub fn set_outputs(&mut self, outputs: Vec<usize>) {
            self.region.outputs = outputs;
        }

        pub fn snapshot(&self) -> Region<C, O> {
            self.region.clone()
        }
    }

    #[derive(Clone)]
    pub struct ProgramBuilder<C, O> {
        regions: Vec<Region<C, O>>,
        /// Callee interning table keyed by live `Rc` pointer identity; `kept` holds the interned sources alive so
        /// the addresses stay valid and unique for the builder's lifetime.
        interned_callees: HashMap<*const (), RegionId>,
        kept: Vec<Rc<Program<C, O>>>,
    }

    impl<C: Clone, O: Clone> ProgramBuilder<C, O> {
        pub fn new() -> Self {
            Self { regions: Vec::new(), interned_callees: HashMap::new(), kept: Vec::new() }
        }

        /// Consumes a region builder into a sealed, attachable region.
        pub fn seal(&mut self, builder: RegionBuilder<C, O>) -> RegionId {
            self.seal_region(builder.region)
        }

        fn seal_region(&mut self, region: Region<C, O>) -> RegionId {
            let id = RegionId(self.regions.len());
            self.regions.push(region);
            id
        }

        /// Imports the complete reachable closure of `source`'s region `root` (lexical descendants, referenced
        /// callees, and their transitive closures), preserving sharing within the imported closure via `remap`.
        pub fn add_region_closure(
            &mut self,
            source: &Program<C, O>,
            root: RegionId,
            remap: &mut HashMap<RegionId, RegionId>,
        ) -> RegionId {
            if let Some(mapped) = remap.get(&root) {
                return *mapped;
            }
            // Reserve the slot first so cyclic-looking self-references would be caught rather than recursing.
            let id = RegionId(self.regions.len());
            self.regions.push(Region {
                atoms: Vec::new(),
                instructions: Vec::new(),
                inputs: Vec::new(),
                outputs: Vec::new(),
            });
            remap.insert(root, id);
            let mut region = source.regions[root.0].clone();
            for instruction in &mut region.instructions {
                for attached in &mut instruction.regions {
                    *attached = self.add_region_closure(source, *attached, remap);
                }
            }
            self.regions[id.0] = region;
            id
        }

        /// Imports a whole program as an owned lexical subtree (never interned: two imports = two subtrees).
        pub fn import_region(&mut self, source: &Program<C, O>) -> RegionId {
            self.add_region_closure(source, source.entry, &mut HashMap::new())
        }

        /// Imports a program as a shareable callee root, interning by live `Rc` identity within this builder.
        pub fn intern_callee(&mut self, source: &Rc<Program<C, O>>) -> RegionId {
            let key = Rc::as_ptr(source) as *const ();
            if let Some(id) = self.interned_callees.get(&key) {
                return *id;
            }
            let id = self.add_region_closure(source, source.entry, &mut HashMap::new());
            self.interned_callees.insert(key, id);
            self.kept.push(source.clone());
            id
        }

        pub fn build(self, entry: RegionId) -> Program<C, O> {
            Program { regions: self.regions, entry }
        }
    }

    // ------------------------------------------------------------------------------------------------------------
    // Operations: non-recursive payloads and region provenance.
    // ------------------------------------------------------------------------------------------------------------

    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    pub struct OutputRegionProvenance {
        pub region_index: usize,
        pub output_index: usize,
    }

    pub trait Operation: Clone + 'static {
        fn name(&self) -> &'static str;
        fn output_count(&self) -> usize;
        fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
            let _ = output_index;
            Vec::new()
        }
    }

    /// Primitive arithmetic: the "large primitive family" stand-in.
    #[derive(Copy, Clone, Debug, PartialEq)]
    pub enum Prim {
        Add,
        Mul,
    }

    /// The higher-order stand-in. CRUCIALLY it owns no program and does not mention the operation enum: two
    /// lexical regions (`true`, `false`) live on the instruction. Payload is metadata only.
    #[derive(Copy, Clone, Debug, PartialEq)]
    pub struct Cond;

    /// The operation enum. No `Box`, no `Self`-mentioning variant, no recursive type equation.
    #[derive(Clone, Debug, PartialEq)]
    pub enum Op {
        Prim(Prim),
        Cond(Cond),
    }

    impl Operation for Prim {
        fn name(&self) -> &'static str {
            match self {
                Prim::Add => "add",
                Prim::Mul => "mul",
            }
        }
        fn output_count(&self) -> usize {
            1
        }
    }

    impl Operation for Cond {
        fn name(&self) -> &'static str {
            "cond"
        }
        fn output_count(&self) -> usize {
            1
        }
        fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
            vec![
                OutputRegionProvenance { region_index: 0, output_index },
                OutputRegionProvenance { region_index: 1, output_index },
            ]
        }
    }

    impl Operation for Op {
        fn name(&self) -> &'static str {
            match self {
                Op::Prim(operation) => operation.name(),
                Op::Cond(operation) => operation.name(),
            }
        }
        fn output_count(&self) -> usize {
            match self {
                Op::Prim(operation) => operation.output_count(),
                Op::Cond(operation) => operation.output_count(),
            }
        }
        fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
            match self {
                Op::Prim(operation) => operation.output_region_provenance(output_index),
                Op::Cond(operation) => operation.output_region_provenance(output_index),
            }
        }
    }

    impl From<Prim> for Op {
        fn from(operation: Prim) -> Self {
            Op::Prim(operation)
        }
    }

    impl From<Cond> for Op {
        fn from(operation: Cond) -> Self {
            Op::Cond(operation)
        }
    }

    // ------------------------------------------------------------------------------------------------------------
    // Contexts, mirroring the production `Context` protocol plus the attachment-aware hook.
    // ------------------------------------------------------------------------------------------------------------

    pub trait Context {
        type Value: Clone;
        type Operation: Operation;

        /// Binds `operation` over `inputs`, together with any freshly authored nested computations (`regions` and
        /// `callees`, in the operation-defined order); most operations carry neither and pass `&[]` for both.
        fn bind(
            &self,
            operation: Self::Operation,
            regions: &[Program<Stored, Self::Operation>],
            callees: &[Rc<Program<Stored, Self::Operation>>],
            inputs: &[Self::Value],
        ) -> R<Vec<Self::Value>>;
    }

    /// Eager context: runtime values are `f64` (distinct from `Stored`), captures substituted from a table.
    #[derive(Clone)]
    pub struct EagerContext {
        pub captures: Rc<Vec<f64>>,
    }

    impl EagerContext {
        pub fn lift(&self, constant: &Stored) -> R<f64> {
            match constant {
                Stored::Literal(value) => Ok(*value),
                Stored::Capture(index) => {
                    self.captures.get(*index).copied().ok_or_else(|| format!("missing capture {index}"))
                }
            }
        }
    }

    impl Context for EagerContext {
        type Value = f64;
        type Operation = Op;

        fn bind(
            &self,
            operation: Op,
            regions: &[Program<Stored, Op>],
            callees: &[Rc<Program<Stored, Op>>],
            inputs: &[f64],
        ) -> R<Vec<f64>> {
            // Eager contexts interpret freshly authored nested computations through the detached access path. The
            // same empty detached driver serves region-free applications, whose rules simply do not invoke it.
            let detached = DetachedInterpret { context: self, regions, callees };
            operation.interpret(self, &detached, inputs)
        }
    }

    /// Staging context: values are atom ids in a shared builder (a minimal stand-in for tracers).
    #[derive(Clone)]
    pub struct TracingContext {
        pub builder: Rc<RefCell<RegionBuilder<Stored, Op>>>,
        pub program: Rc<RefCell<ProgramBuilder<Stored, Op>>>,
    }

    impl TracingContext {
        pub fn new() -> Self {
            Self {
                builder: Rc::new(RefCell::new(RegionBuilder::new())),
                program: Rc::new(RefCell::new(ProgramBuilder::new())),
            }
        }

        /// Snapshots the traced region into a finished program. Takes `&self` because transform drivers hand
        /// clones of the context to rules (mirroring production tracer-carried contexts).
        pub fn finish(&self, outputs: Vec<usize>) -> Program<Stored, Op> {
            let mut region = self.builder.borrow().snapshot();
            region.outputs = outputs;
            let mut program_builder = self.program.borrow().clone();
            let entry = program_builder.seal_region(region);
            program_builder.build(entry)
        }
    }

    impl Context for TracingContext {
        type Value = usize;
        type Operation = Op;

        fn bind(
            &self,
            operation: Op,
            regions: &[Program<Stored, Op>],
            callees: &[Rc<Program<Stored, Op>>],
            inputs: &[usize],
        ) -> R<Vec<usize>> {
            // Staging contexts add the nested computations to the destination program and attach their ids. Owned
            // regions import through the copying policy and callees through the interning policy, but the resulting
            // instruction carries one ordered region list (owned regions first, callee regions after).
            let mut program = self.program.borrow_mut();
            let mut attached = regions.iter().map(|region| program.import_region(region)).collect::<Vec<_>>();
            attached.extend(callees.iter().map(|callee| program.intern_callee(callee)));
            let count = operation.output_count();
            Ok(self.builder.borrow_mut().add_instruction(operation, inputs.to_vec(), count, attached))
        }
    }
    // ------------------------------------------------------------------------------------------------------------
    // Call-scoped region drivers (invariant 3): public contracts and model-private concrete implementations.
    // ------------------------------------------------------------------------------------------------------------

    pub trait InterpretationDriver<V> {
        fn interpret_region(&self, index: usize, inputs: Vec<V>) -> R<Vec<V>>;
    }

    /// Detached access to freshly authored nested computations, used by [`EagerContext::bind`].
    struct DetachedInterpret<'a> {
        context: &'a EagerContext,
        regions: &'a [Program<Stored, Op>],
        callees: &'a [Rc<Program<Stored, Op>>],
    }

    impl InterpretationDriver<f64> for DetachedInterpret<'_> {
        fn interpret_region(&self, index: usize, inputs: Vec<f64>) -> R<Vec<f64>> {
            let program = if let Some(region) = self.regions.get(index) {
                region
            } else {
                self.callees
                    .get(index - self.regions.len())
                    .map(Rc::as_ref)
                    .ok_or_else(|| "operation has no attached regions".to_string())?
            };
            program.interpret_in_context(self.context, &|constant| self.context.lift(constant), inputs)
        }
    }

    /// Replay access: resolves the active instruction's regions against the source arena and recurses through
    /// the driver. Constructed only by [`Program::interpret_in_context`], where the family bound already holds.
    struct ReplayInterpret<'a, C, V, Ctx, O> {
        program: &'a Program<C, O>,
        instruction: &'a Instruction<O>,
        context: &'a Ctx,
        lift: &'a dyn Fn(&C) -> R<V>,
    }

    impl<C: Clone, V: Clone, Ctx, O> InterpretationDriver<V> for ReplayInterpret<'_, C, V, Ctx, O>
    where
        O: InterpretableOperation<V, Ctx>,
    {
        fn interpret_region(&self, index: usize, inputs: Vec<V>) -> R<Vec<V>> {
            let region = self.instruction.regions[index];
            self.program.interpret_region_in_context(region, self.context, self.lift, inputs)
        }
    }

    // ------------------------------------------------------------------------------------------------------------
    // Interpretation: single `interpret` method, deliberately unbounded `C`, and direct per-variant bounds.
    // ------------------------------------------------------------------------------------------------------------

    pub trait InterpretableOperation<V, C>: Operation {
        fn interpret<D: InterpretationDriver<V>>(&self, context: &C, driver: &D, inputs: &[V]) -> R<Vec<V>>;
    }

    impl<C> InterpretableOperation<f64, C> for Prim {
        fn interpret<D: InterpretationDriver<f64>>(&self, _context: &C, _driver: &D, inputs: &[f64]) -> R<Vec<f64>> {
            match self {
                Prim::Add => Ok(vec![inputs[0] + inputs[1]]),
                Prim::Mul => Ok(vec![inputs[0] * inputs[1]]),
            }
        }
    }

    impl<C> InterpretableOperation<f64, C> for Cond {
        fn interpret<D: InterpretationDriver<f64>>(&self, _context: &C, driver: &D, inputs: &[f64]) -> R<Vec<f64>> {
            // Predicate first, branch operands after; nonzero selects the `true` region.
            let index = if inputs[0] != 0.0 { 0 } else { 1 };
            driver.interpret_region(index, inputs[1..].to_vec())
        }
    }

    /// The enum dispatch uses direct, finite per-variant bounds because no variant payload mentions `Op`.
    impl<V, C> InterpretableOperation<V, C> for Op
    where
        Prim: InterpretableOperation<V, C>,
        Cond: InterpretableOperation<V, C>,
    {
        fn interpret<D: InterpretationDriver<V>>(&self, context: &C, driver: &D, inputs: &[V]) -> R<Vec<V>> {
            match self {
                Op::Prim(operation) => operation.interpret(context, driver, inputs),
                Op::Cond(operation) => operation.interpret(context, driver, inputs),
            }
        }
    }

    impl<C: Clone, O: Operation> Program<C, O> {
        pub fn interpret_in_context<V: Clone, Ctx>(
            &self,
            context: &Ctx,
            lift: &dyn Fn(&C) -> R<V>,
            inputs: Vec<V>,
        ) -> R<Vec<V>>
        where
            O: InterpretableOperation<V, Ctx>,
        {
            self.interpret_region_in_context(self.entry, context, lift, inputs)
        }

        fn interpret_region_in_context<V: Clone, Ctx>(
            &self,
            region: RegionId,
            context: &Ctx,
            lift: &dyn Fn(&C) -> R<V>,
            inputs: Vec<V>,
        ) -> R<Vec<V>>
        where
            O: InterpretableOperation<V, Ctx>,
        {
            let region_data = &self.regions[region.0];
            let mut values: Vec<Option<V>> = vec![None; region_data.atoms.len()];
            for (input, value) in region_data.inputs.iter().zip(inputs) {
                values[*input] = Some(value);
            }
            for (index, atom) in region_data.atoms.iter().enumerate() {
                if let Atom::Constant(constant) = atom {
                    values[index] = Some(lift(constant)?);
                }
            }
            for instruction in &region_data.instructions {
                let operands = instruction
                    .inputs
                    .iter()
                    .map(|input| values[*input].clone().ok_or_else(|| "unbound operand".to_string()))
                    .collect::<R<Vec<_>>>()?;
                let replay = ReplayInterpret { program: self, instruction, context, lift };
                let outputs = instruction.operation.interpret(context, &replay, &operands)?;
                for (output, value) in instruction.outputs.iter().zip(outputs) {
                    values[*output] = Some(value);
                }
            }
            region_data
                .outputs
                .iter()
                .map(|output| values[*output].clone().ok_or_else(|| "unbound output".to_string()))
                .collect()
        }
    }

    // ------------------------------------------------------------------------------------------------------------
    // Partial evaluation: fixed known-side context `C`, one semantic method, and direct driver access.
    // ------------------------------------------------------------------------------------------------------------

    #[derive(Clone, Debug)]
    pub enum PartialValue<V> {
        Known(V),
        Unknown,
    }

    impl<V> PartialValue<V> {
        pub fn known(&self) -> Option<&V> {
            match self {
                PartialValue::Known(value) => Some(value),
                PartialValue::Unknown => None,
            }
        }
    }

    pub trait PartialEvaluationDriver<V> {
        fn partially_evaluate_region(&self, index: usize, inputs: Vec<PartialValue<V>>) -> R<Vec<PartialValue<V>>>;
    }

    pub trait PartiallyEvaluatableOperation<C: Context>: Operation {
        fn partially_evaluate<D: PartialEvaluationDriver<C::Value>>(
            &self,
            context: &C,
            driver: &D,
            inputs: &[PartialValue<C::Value>],
        ) -> R<Vec<PartialValue<C::Value>>>;
    }

    impl<C: Context<Operation = Op>> PartiallyEvaluatableOperation<C> for Prim {
        fn partially_evaluate<D: PartialEvaluationDriver<C::Value>>(
            &self,
            context: &C,
            _driver: &D,
            inputs: &[PartialValue<C::Value>],
        ) -> R<Vec<PartialValue<C::Value>>> {
            if inputs.iter().all(|input| input.known().is_some()) {
                let operands = inputs.iter().map(|input| input.known().unwrap().clone()).collect::<Vec<_>>();
                Ok(context.bind(Op::Prim(*self), &[], &[], &operands)?.into_iter().map(PartialValue::Known).collect())
            } else {
                Ok(vec![PartialValue::Unknown; self.output_count()])
            }
        }
    }

    impl<C: Context<Operation = Op>> PartiallyEvaluatableOperation<C> for Cond {
        fn partially_evaluate<D: PartialEvaluationDriver<C::Value>>(
            &self,
            _context: &C,
            driver: &D,
            inputs: &[PartialValue<C::Value>],
        ) -> R<Vec<PartialValue<C::Value>>> {
            // Semantics stub: production selects the region from the known predicate value and residualizes the
            // application (with transformed attachments) when the predicate is unknown. The solver-relevant part is
            // the region re-entry below with no bound on `Op`.
            if inputs[0].known().is_some() {
                driver.partially_evaluate_region(0, inputs[1..].to_vec())
            } else {
                Ok(vec![PartialValue::Unknown; self.output_count()])
            }
        }
    }

    impl<C: Context<Operation = Op>> PartiallyEvaluatableOperation<C> for Op {
        fn partially_evaluate<D: PartialEvaluationDriver<C::Value>>(
            &self,
            context: &C,
            driver: &D,
            inputs: &[PartialValue<C::Value>],
        ) -> R<Vec<PartialValue<C::Value>>> {
            match self {
                Op::Prim(operation) => operation.partially_evaluate(context, driver, inputs),
                Op::Cond(operation) => operation.partially_evaluate(context, driver, inputs),
            }
        }
    }

    struct ReplayPartial<'a, C, Ctx: Context, O> {
        program: &'a Program<C, O>,
        instruction: &'a Instruction<O>,
        context: &'a Ctx,
        lift: &'a dyn Fn(&C) -> R<Ctx::Value>,
    }

    impl<C: Clone, Ctx: Context, O> PartialEvaluationDriver<Ctx::Value> for ReplayPartial<'_, C, Ctx, O>
    where
        O: PartiallyEvaluatableOperation<Ctx>,
    {
        fn partially_evaluate_region(
            &self,
            index: usize,
            inputs: Vec<PartialValue<Ctx::Value>>,
        ) -> R<Vec<PartialValue<Ctx::Value>>> {
            let region = self.instruction.regions[index];
            self.program.partially_evaluate_region_in_context(region, self.context, self.lift, inputs)
        }
    }

    impl<C: Clone, O: Operation> Program<C, O> {
        pub fn partially_evaluate_in_context<Ctx: Context>(
            &self,
            context: &Ctx,
            lift: &dyn Fn(&C) -> R<Ctx::Value>,
            inputs: Vec<PartialValue<Ctx::Value>>,
        ) -> R<Vec<PartialValue<Ctx::Value>>>
        where
            O: PartiallyEvaluatableOperation<Ctx>,
        {
            self.partially_evaluate_region_in_context(self.entry, context, lift, inputs)
        }

        fn partially_evaluate_region_in_context<Ctx: Context>(
            &self,
            region: RegionId,
            context: &Ctx,
            lift: &dyn Fn(&C) -> R<Ctx::Value>,
            inputs: Vec<PartialValue<Ctx::Value>>,
        ) -> R<Vec<PartialValue<Ctx::Value>>>
        where
            O: PartiallyEvaluatableOperation<Ctx>,
        {
            let region_data = &self.regions[region.0];
            let mut values: Vec<Option<PartialValue<Ctx::Value>>> = vec![None; region_data.atoms.len()];
            for (input, value) in region_data.inputs.iter().zip(inputs) {
                values[*input] = Some(value);
            }
            for (index, atom) in region_data.atoms.iter().enumerate() {
                if let Atom::Constant(constant) = atom {
                    values[index] = Some(PartialValue::Known(lift(constant)?));
                }
            }
            for instruction in &region_data.instructions {
                let operands = instruction
                    .inputs
                    .iter()
                    .map(|input| values[*input].clone().ok_or_else(|| "unbound operand".to_string()))
                    .collect::<R<Vec<_>>>()?;
                let replay = ReplayPartial { program: self, instruction, context, lift };
                let outputs = instruction.operation.partially_evaluate(context, &replay, &operands)?;
                for (output, value) in instruction.outputs.iter().zip(outputs) {
                    values[*output] = Some(value);
                }
            }
            region_data
                .outputs
                .iter()
                .map(|output| values[*output].clone().ok_or_else(|| "unbound output".to_string()))
                .collect()
        }
    }

    // ------------------------------------------------------------------------------------------------------------
    // Transform context stacks, mirroring the fresh-context driver shapes that diverge today. Semantics are stubs;
    // the bounds, stacks, and borrow shapes are the gate items.
    // ------------------------------------------------------------------------------------------------------------

    pub struct BatchingContext<P: Context> {
        pub parent: P,
    }

    impl<P: Context> Context for BatchingContext<P> {
        type Value = P::Value;
        type Operation = P::Operation;

        fn bind(
            &self,
            operation: P::Operation,
            regions: &[Program<Stored, P::Operation>],
            callees: &[Rc<Program<Stored, P::Operation>>],
            inputs: &[P::Value],
        ) -> R<Vec<P::Value>> {
            self.parent.bind(operation, regions, callees, inputs)
        }
    }

    pub struct DifferentiationContext<P: Context> {
        pub parent: P,
    }

    impl<P: Context> Context for DifferentiationContext<P> {
        type Value = P::Value;
        type Operation = P::Operation;

        fn bind(
            &self,
            operation: P::Operation,
            regions: &[Program<Stored, P::Operation>],
            callees: &[Rc<Program<Stored, P::Operation>>],
            inputs: &[P::Value],
        ) -> R<Vec<P::Value>> {
            self.parent.bind(operation, regions, callees, inputs)
        }
    }

    /// Known-side partial evaluation over a parent context: all-known applications fold to the parent, anything
    /// else is unknown. Mirrors the composed `DifferentiationContext<PartialEvaluationContext<TracingContext>>`
    /// stack that `Program::linearize` uses in production.
    pub struct PartialEvaluationContext<P: Context> {
        pub parent: P,
    }

    impl<P: Context> Context for PartialEvaluationContext<P> {
        type Value = PartialValue<P::Value>;
        type Operation = P::Operation;

        fn bind(
            &self,
            operation: P::Operation,
            regions: &[Program<Stored, P::Operation>],
            callees: &[Rc<Program<Stored, P::Operation>>],
            inputs: &[PartialValue<P::Value>],
        ) -> R<Vec<PartialValue<P::Value>>> {
            if inputs.iter().all(|input| input.known().is_some()) {
                let operands = inputs.iter().map(|input| input.known().unwrap().clone()).collect::<Vec<_>>();
                let outputs = self.parent.bind(operation, regions, callees, &operands)?;
                Ok(outputs.into_iter().map(PartialValue::Known).collect())
            } else {
                Ok(vec![PartialValue::Unknown; operation.output_count()])
            }
        }
    }

    // ------------------------------------------------------------------------------------------------------------
    // Batching: fresh `BatchingContext<TracingContext>` stack and instruction-scoped region access.
    // ------------------------------------------------------------------------------------------------------------

    pub trait BatchingDriver<O> {
        fn region(&self, index: usize) -> R<RegionRef<'_, Stored, O>>;

        fn batch_program(&self, region: RegionRef<'_, Stored, O>) -> R<Program<Stored, O>>;
    }

    pub trait BatchableOperation<C: Context>: Operation {
        fn batch<D: BatchingDriver<C::Operation>>(
            &self,
            context: &C,
            driver: &D,
            inputs: &[C::Value],
        ) -> R<Vec<C::Value>>;
    }

    impl<C: Context> BatchableOperation<C> for Prim
    where
        C::Operation: From<Prim>,
    {
        fn batch<D: BatchingDriver<C::Operation>>(
            &self,
            context: &C,
            _driver: &D,
            inputs: &[C::Value],
        ) -> R<Vec<C::Value>> {
            context.bind((*self).into(), &[], &[], inputs)
        }
    }

    impl<C: Context> BatchableOperation<C> for Cond
    where
        C::Operation: From<Cond>,
    {
        fn batch<D: BatchingDriver<C::Operation>>(
            &self,
            context: &C,
            driver: &D,
            inputs: &[C::Value],
        ) -> R<Vec<C::Value>> {
            // The higher-order rule requests transformed regions through its driver and emits one
            // destination application with attachments; it never restates a bound on the operation enum.
            let true_region = driver.batch_program(driver.region(0)?)?;
            let false_region = driver.batch_program(driver.region(1)?)?;
            context.bind((*self).into(), &[true_region, false_region], &[], inputs)
        }
    }

    impl<C: Context> BatchableOperation<C> for Op
    where
        Prim: BatchableOperation<C>,
        Cond: BatchableOperation<C>,
    {
        fn batch<D: BatchingDriver<C::Operation>>(
            &self,
            context: &C,
            driver: &D,
            inputs: &[C::Value],
        ) -> R<Vec<C::Value>> {
            match self {
                Op::Prim(operation) => operation.batch(context, driver, inputs),
                Op::Cond(operation) => operation.batch(context, driver, inputs),
            }
        }
    }

    struct ReplayBatch<'a> {
        program: &'a Program<Stored, Op>,
        instruction: &'a Instruction<Op>,
    }

    impl BatchingDriver<Op> for ReplayBatch<'_> {
        fn region(&self, index: usize) -> R<RegionRef<'_, Stored, Op>> {
            Ok(self.program.region_ref(self.instruction.regions[index]))
        }

        fn batch_program(&self, region: RegionRef<'_, Stored, Op>) -> R<Program<Stored, Op>> {
            // Runtime recursion through the driver: the bound is re-established here by the concrete entry point,
            // never by the rule that asked.
            region.into_program().batched()
        }
    }

    impl Program<Stored, Op> {
        /// Mirrors `Program::batched`: fresh `BatchingContext<TracingContext>` stack. Proving
        /// `Op: BatchableOperation<BatchingContext<TracingContext>>` here is finite because no `Op` variant payload
        /// mentions `Op`.
        pub fn batched(&self) -> R<Program<Stored, Op>> {
            let tracing = TracingContext::new();
            let context = BatchingContext { parent: tracing.clone() };
            let entry = self.entry_region();
            let inputs = entry.inputs.iter().map(|_| tracing.builder.borrow_mut().add_input()).collect::<Vec<_>>();
            let outputs = self.replay_batched(self.entry, &context, inputs)?;
            Ok(tracing.finish(outputs))
        }

        fn replay_batched(
            &self,
            region: RegionId,
            context: &BatchingContext<TracingContext>,
            inputs: Vec<usize>,
        ) -> R<Vec<usize>> {
            let region_data = &self.regions[region.0];
            let mut values: Vec<Option<usize>> = vec![None; region_data.atoms.len()];
            for (input, value) in region_data.inputs.iter().zip(inputs) {
                values[*input] = Some(value);
            }
            for (index, atom) in region_data.atoms.iter().enumerate() {
                if let Atom::Constant(constant) = atom {
                    values[index] = Some(context.parent.builder.borrow_mut().add_constant(constant.clone()));
                }
            }
            for instruction in &region_data.instructions {
                let operands = instruction
                    .inputs
                    .iter()
                    .map(|input| values[*input].ok_or_else(|| "unbound operand".to_string()))
                    .collect::<R<Vec<_>>>()?;
                let replay = ReplayBatch { program: self, instruction };
                let outputs = instruction.operation.batch(context, &replay, &operands)?;
                for (output, value) in instruction.outputs.iter().zip(outputs) {
                    values[*output] = Some(value);
                }
            }
            region_data
                .outputs
                .iter()
                .map(|output| values[*output].ok_or_else(|| "unbound output".to_string()))
                .collect()
        }
    }

    // ------------------------------------------------------------------------------------------------------------
    // Differentiation: one `jvp` rule serves both the fused-JVP driver and the composed linearize driver, matching
    // the production single-method contract.
    // ------------------------------------------------------------------------------------------------------------

    #[derive(Clone, Debug)]
    pub struct Dual<V> {
        pub primal: V,
        pub tangent: V,
    }

    pub trait DifferentiationDriver<O> {
        fn region(&self, index: usize) -> R<RegionRef<'_, Stored, O>>;

        fn jvp_program(&self, region: RegionRef<'_, Stored, O>) -> R<Program<Stored, O>>;
    }

    pub trait DifferentiableOperation<C: Context>: Operation {
        fn jvp<D: DifferentiationDriver<C::Operation>>(
            &self,
            context: &C,
            driver: &D,
            inputs: &[Dual<C::Value>],
        ) -> R<Vec<Dual<C::Value>>>;
    }

    impl<C: Context> DifferentiableOperation<C> for Prim
    where
        C::Operation: From<Prim>,
    {
        fn jvp<D: DifferentiationDriver<C::Operation>>(
            &self,
            context: &C,
            _driver: &D,
            inputs: &[Dual<C::Value>],
        ) -> R<Vec<Dual<C::Value>>> {
            let primals = inputs.iter().map(|input| input.primal.clone()).collect::<Vec<_>>();
            let tangents = inputs.iter().map(|input| input.tangent.clone()).collect::<Vec<_>>();
            let primal = context.bind((*self).into(), &[], &[], &primals)?.remove(0);
            let tangent = match self {
                // d(a + b) = da + db
                Prim::Add => context.bind(Prim::Add.into(), &[], &[], &tangents)?.remove(0),
                // d(a * b) = da * b + a * db
                Prim::Mul => {
                    let left =
                        context.bind(Prim::Mul.into(), &[], &[], &[tangents[0].clone(), primals[1].clone()])?.remove(0);
                    let right =
                        context.bind(Prim::Mul.into(), &[], &[], &[primals[0].clone(), tangents[1].clone()])?.remove(0);
                    context.bind(Prim::Add.into(), &[], &[], &[left, right])?.remove(0)
                }
            };
            Ok(vec![Dual { primal, tangent }])
        }
    }

    impl<C: Context> DifferentiableOperation<C> for Cond
    where
        C::Operation: From<Cond>,
    {
        fn jvp<D: DifferentiationDriver<C::Operation>>(
            &self,
            context: &C,
            driver: &D,
            inputs: &[Dual<C::Value>],
        ) -> R<Vec<Dual<C::Value>>> {
            // Semantics stub: production emits one primal condition and one tangent condition over the jvp'd
            // branches with proper operand wiring. The solver-relevant parts are the driver-based region re-entry and
            // attachment-aware emission with no bound on the operation enum.
            let true_region = driver.jvp_program(driver.region(0)?)?;
            let false_region = driver.jvp_program(driver.region(1)?)?;
            let primals = inputs.iter().map(|input| input.primal.clone()).collect::<Vec<_>>();
            let tangents = inputs.iter().map(|input| input.tangent.clone()).collect::<Vec<_>>();
            let primal =
                context.bind((*self).into(), &[true_region.clone(), false_region.clone()], &[], &primals)?.remove(0);
            let tangent = context.bind((*self).into(), &[true_region, false_region], &[], &tangents)?.remove(0);
            Ok(vec![Dual { primal, tangent }])
        }
    }

    impl<C: Context> DifferentiableOperation<C> for Op
    where
        Prim: DifferentiableOperation<C>,
        Cond: DifferentiableOperation<C>,
    {
        fn jvp<D: DifferentiationDriver<C::Operation>>(
            &self,
            context: &C,
            driver: &D,
            inputs: &[Dual<C::Value>],
        ) -> R<Vec<Dual<C::Value>>> {
            match self {
                Op::Prim(operation) => operation.jvp(context, driver, inputs),
                Op::Cond(operation) => operation.jvp(context, driver, inputs),
            }
        }
    }

    struct ReplayJvp<'a> {
        program: &'a Program<Stored, Op>,
        instruction: &'a Instruction<Op>,
    }

    impl DifferentiationDriver<Op> for ReplayJvp<'_> {
        fn region(&self, index: usize) -> R<RegionRef<'_, Stored, Op>> {
            Ok(self.program.region_ref(self.instruction.regions[index]))
        }

        fn jvp_program(&self, region: RegionRef<'_, Stored, Op>) -> R<Program<Stored, Op>> {
            region.into_program().jvp()
        }
    }

    struct ReplayLinearize<'a> {
        program: &'a Program<Stored, Op>,
        instruction: &'a Instruction<Op>,
    }

    impl DifferentiationDriver<Op> for ReplayLinearize<'_> {
        fn region(&self, index: usize) -> R<RegionRef<'_, Stored, Op>> {
            Ok(self.program.region_ref(self.instruction.regions[index]))
        }

        fn jvp_program(&self, region: RegionRef<'_, Stored, Op>) -> R<Program<Stored, Op>> {
            // The linearize driver hands rules linearized regions through the same driver contract: which transform
            // runs is the driver's choice, invisible to the rule.
            Ok(region.into_program().linearize()?.0)
        }
    }

    impl Program<Stored, Op> {
        /// Mirrors `Program::jvp`: fresh `DifferentiationContext<TracingContext>` stack.
        pub fn jvp(&self) -> R<Program<Stored, Op>> {
            let tracing = TracingContext::new();
            let context = DifferentiationContext { parent: tracing.clone() };
            let entry = self.entry_region();
            let primal_inputs =
                entry.inputs.iter().map(|_| tracing.builder.borrow_mut().add_input()).collect::<Vec<_>>();
            let tangent_inputs =
                entry.inputs.iter().map(|_| tracing.builder.borrow_mut().add_input()).collect::<Vec<_>>();
            let duals = primal_inputs
                .into_iter()
                .zip(tangent_inputs)
                .map(|(primal, tangent)| Dual { primal, tangent })
                .collect::<Vec<_>>();
            let outputs = self.replay_jvp(self.entry, &context, duals, &|constant, tracing| {
                let primal = tracing.builder.borrow_mut().add_constant(constant.clone());
                let tangent = tracing.builder.borrow_mut().add_constant(Stored::Literal(0.0));
                Dual { primal, tangent }
            })?;
            let mut flat = outputs.iter().map(|output| output.primal).collect::<Vec<_>>();
            flat.extend(outputs.iter().map(|output| output.tangent));
            Ok(tracing.finish(flat))
        }

        fn replay_jvp(
            &self,
            region: RegionId,
            context: &DifferentiationContext<TracingContext>,
            inputs: Vec<Dual<usize>>,
            constant: &dyn Fn(&Stored, &TracingContext) -> Dual<usize>,
        ) -> R<Vec<Dual<usize>>> {
            let region_data = &self.regions[region.0];
            let mut values: Vec<Option<Dual<usize>>> = vec![None; region_data.atoms.len()];
            for (input, value) in region_data.inputs.iter().zip(inputs) {
                values[*input] = Some(value);
            }
            for (index, atom) in region_data.atoms.iter().enumerate() {
                if let Atom::Constant(stored) = atom {
                    values[index] = Some(constant(stored, &context.parent));
                }
            }
            for instruction in &region_data.instructions {
                let operands = instruction
                    .inputs
                    .iter()
                    .map(|input| values[*input].clone().ok_or_else(|| "unbound operand".to_string()))
                    .collect::<R<Vec<_>>>()?;
                let replay = ReplayJvp { program: self, instruction };
                let outputs = instruction.operation.jvp(context, &replay, &operands)?;
                for (output, value) in instruction.outputs.iter().zip(outputs) {
                    values[*output] = Some(value);
                }
            }
            region_data
                .outputs
                .iter()
                .map(|output| values[*output].clone().ok_or_else(|| "unbound output".to_string()))
                .collect()
        }

        /// Mirrors `Program::linearize`: the COMPOSED `DifferentiationContext<PartialEvaluationContext<
        /// TracingContext>>` stack, driven through the same single `jvp` rule. Returns a stub primal program plus a
        /// stub residual count; the compile of this composed bound is the Phase 0 gate item.
        pub fn linearize(&self) -> R<(Program<Stored, Op>, usize)> {
            let tracing = TracingContext::new();
            let partial = PartialEvaluationContext { parent: tracing.clone() };
            let context = DifferentiationContext { parent: partial };
            let entry = self.entry_region();
            let duals = entry
                .inputs
                .iter()
                .map(|_| Dual {
                    primal: PartialValue::Known(tracing.builder.borrow_mut().add_input()),
                    tangent: PartialValue::Unknown,
                })
                .collect::<Vec<_>>();
            let outputs = self.replay_linearize(self.entry, &context, duals)?;
            let primal_outputs = outputs
                .iter()
                .map(|output| output.primal.known().copied().ok_or_else(|| "unknown primal output".to_string()))
                .collect::<R<Vec<_>>>()?;
            Ok((tracing.finish(primal_outputs), 0))
        }

        fn replay_linearize(
            &self,
            region: RegionId,
            context: &DifferentiationContext<PartialEvaluationContext<TracingContext>>,
            inputs: Vec<Dual<PartialValue<usize>>>,
        ) -> R<Vec<Dual<PartialValue<usize>>>> {
            let region_data = &self.regions[region.0];
            let mut values: Vec<Option<Dual<PartialValue<usize>>>> = vec![None; region_data.atoms.len()];
            for (input, value) in region_data.inputs.iter().zip(inputs) {
                values[*input] = Some(value);
            }
            for (index, atom) in region_data.atoms.iter().enumerate() {
                if let Atom::Constant(stored) = atom {
                    let tracer = context.parent.parent.builder.borrow_mut().add_constant(stored.clone());
                    values[index] = Some(Dual { primal: PartialValue::Known(tracer), tangent: PartialValue::Unknown });
                }
            }
            for instruction in &region_data.instructions {
                let operands = instruction
                    .inputs
                    .iter()
                    .map(|input| values[*input].clone().ok_or_else(|| "unbound operand".to_string()))
                    .collect::<R<Vec<_>>>()?;
                let replay = ReplayLinearize { program: self, instruction };
                let outputs = instruction.operation.jvp(context, &replay, &operands)?;
                for (output, value) in instruction.outputs.iter().zip(outputs) {
                    values[*output] = Some(value);
                }
            }
            region_data
                .outputs
                .iter()
                .map(|output| values[*output].clone().ok_or_else(|| "unbound output".to_string()))
                .collect()
        }
    }

    // ------------------------------------------------------------------------------------------------------------
    // Transposition: `&mut` destination context coexisting with borrowed source-region access.
    // ------------------------------------------------------------------------------------------------------------

    pub trait TranspositionDriver<O> {
        fn region(&self, index: usize) -> R<RegionRef<'_, Stored, O>>;

        fn transpose_program(&self, region: RegionRef<'_, Stored, O>) -> R<Program<Stored, O>>;
    }

    pub trait TransposableOperation: Operation {
        /// Mirrors the production `&mut TracingContext` destination: the driver borrows the source program
        /// immutably while the rule stages into the mutable destination, with no aliasing between them.
        fn transpose<D: TranspositionDriver<Op>>(
            &self,
            context: &mut TracingContext,
            driver: &D,
            cotangents: &[usize],
        ) -> R<Vec<usize>>;
    }

    impl TransposableOperation for Prim {
        fn transpose<D: TranspositionDriver<Op>>(
            &self,
            context: &mut TracingContext,
            _driver: &D,
            cotangents: &[usize],
        ) -> R<Vec<usize>> {
            // Semantics stub: re-stage the operation on the cotangents.
            context.bind(Op::Prim(*self), &[], &[], cotangents)
        }
    }

    impl TransposableOperation for Cond {
        fn transpose<D: TranspositionDriver<Op>>(
            &self,
            context: &mut TracingContext,
            driver: &D,
            cotangents: &[usize],
        ) -> R<Vec<usize>> {
            let true_region = driver.transpose_program(driver.region(0)?)?;
            let false_region = driver.transpose_program(driver.region(1)?)?;
            context.bind(Op::Cond(*self), &[true_region, false_region], &[], cotangents)
        }
    }

    impl TransposableOperation for Op {
        fn transpose<D: TranspositionDriver<Op>>(
            &self,
            context: &mut TracingContext,
            driver: &D,
            cotangents: &[usize],
        ) -> R<Vec<usize>> {
            match self {
                Op::Prim(operation) => operation.transpose(context, driver, cotangents),
                Op::Cond(operation) => operation.transpose(context, driver, cotangents),
            }
        }
    }

    struct ReplayTranspose<'a> {
        program: &'a Program<Stored, Op>,
        instruction: &'a Instruction<Op>,
    }

    impl TranspositionDriver<Op> for ReplayTranspose<'_> {
        fn region(&self, index: usize) -> R<RegionRef<'_, Stored, Op>> {
            Ok(self.program.region_ref(self.instruction.regions[index]))
        }

        fn transpose_program(&self, region: RegionRef<'_, Stored, Op>) -> R<Program<Stored, Op>> {
            region.into_program().transposed()
        }
    }

    impl Program<Stored, Op> {
        /// Semantics stub of `Program::transpose_with_respect_to`: forward replay standing in for the reverse walk;
        /// the borrow shape (`&mut` destination + immutable source views + a borrowed driver) is the gate item.
        pub fn transposed(&self) -> R<Program<Stored, Op>> {
            let mut context = TracingContext::new();
            let entry = self.entry_region();
            let inputs = entry.inputs.iter().map(|_| context.builder.borrow_mut().add_input()).collect::<Vec<_>>();
            let mut values: Vec<Option<usize>> = vec![None; entry.atoms.len()];
            for (input, value) in entry.inputs.iter().zip(inputs) {
                values[*input] = Some(value);
            }
            for (index, atom) in entry.atoms.iter().enumerate() {
                if let Atom::Constant(constant) = atom {
                    values[index] = Some(context.builder.borrow_mut().add_constant(constant.clone()));
                }
            }
            for instruction in &entry.instructions {
                let operands = instruction
                    .inputs
                    .iter()
                    .map(|input| values[*input].ok_or_else(|| "unbound operand".to_string()))
                    .collect::<R<Vec<_>>>()?;
                let replay = ReplayTranspose { program: self, instruction };
                let outputs = instruction.operation.transpose(&mut context, &replay, &operands)?;
                for (output, value) in instruction.outputs.iter().zip(outputs) {
                    values[*output] = Some(value);
                }
            }
            let outputs = entry
                .outputs
                .iter()
                .map(|output| values[*output].ok_or_else(|| "unbound output".to_string()))
                .collect::<R<Vec<_>>>()?;
            Ok(context.finish(outputs))
        }
    }

    // ------------------------------------------------------------------------------------------------------------
    // Lazy residual-origin resolution.
    // ------------------------------------------------------------------------------------------------------------

    impl<C: Clone, O: Operation> Program<C, O> {
        /// Resolves the semantic origins of one value against this frozen program: empty for inputs/constants
        /// (including synthetic padding, whose producer is a constant), the value itself when its provenance is
        /// empty, and the transitive region-output origins otherwise, deduplicated by first occurrence.
        pub fn resolve_origins(&self, value: ValueId) -> R<Vec<ValueId>> {
            let mut origins = Vec::new();
            self.resolve_origins_into(value, &mut origins)?;
            Ok(origins)
        }

        fn resolve_origins_into(&self, value: ValueId, origins: &mut Vec<ValueId>) -> R<()> {
            let region = self.regions.get(value.region.0).ok_or_else(|| "region out of range".to_string())?;
            match region.atoms.get(value.atom) {
                None => return Err("atom out of range".to_string()),
                Some(Atom::Constant(_)) => return Ok(()),
                Some(Atom::Variable) => {}
            }
            if region.inputs.contains(&value.atom) {
                return Ok(());
            }
            let Some((instruction, local_index)) = region.instructions.iter().find_map(|instruction| {
                instruction
                    .outputs
                    .iter()
                    .position(|output| *output == value.atom)
                    .map(|index| (instruction, index))
            }) else {
                return Ok(());
            };
            let provenance = instruction.operation.output_region_provenance(local_index);
            if provenance.is_empty() {
                if !origins.contains(&value) {
                    origins.push(value);
                }
                return Ok(());
            }
            for origin in provenance {
                let target = instruction
                    .regions
                    .get(origin.region_index)
                    .copied()
                    .ok_or_else(|| "output provenance references a region index out of range".to_string())?;
                let target_region = self.regions.get(target.0).ok_or_else(|| "region out of range".to_string())?;
                let atom = *target_region
                    .outputs
                    .get(origin.output_index)
                    .ok_or_else(|| "output provenance references a region output out of range".to_string())?;
                self.resolve_origins_into(ValueId { region: target, atom }, origins)?;
            }
            Ok(())
        }
    }
}

pub use model::*;

/// Builds a program whose entry is `cond(p, x)` with `depth` further conditions nested inside successive `true`
/// branches. The innermost `true` branch computes `x + capture#0`; every `false` branch computes `x * x`. Depth is
/// runtime data: the same `Op` type serves every level, demonstrating that deeper nesting adds no trait obligations.
fn nested_condition_program(depth: usize) -> Program<Stored, Op> {
    fn branches(builder: &mut ProgramBuilder<Stored, Op>, depth: usize) -> (RegionId, RegionId) {
        let mut false_branch = RegionBuilder::new();
        let x = false_branch.add_input();
        let product = false_branch.add_instruction(Op::Prim(Prim::Mul), vec![x, x], 1, Vec::new())[0];
        false_branch.set_outputs(vec![product]);
        let false_id = builder.seal(false_branch);

        let mut true_branch = RegionBuilder::new();
        let x = true_branch.add_input();
        let output = if depth == 0 {
            let capture = true_branch.add_constant(Stored::Capture(0));
            true_branch.add_instruction(Op::Prim(Prim::Add), vec![x, capture], 1, Vec::new())[0]
        } else {
            let (nested_true, nested_false) = branches(builder, depth - 1);
            let predicate = true_branch.add_constant(Stored::Literal(1.0));
            true_branch.add_instruction(Op::Cond(Cond), vec![predicate, x], 1, vec![nested_true, nested_false])[0]
        };
        true_branch.set_outputs(vec![output]);
        (builder.seal(true_branch), false_id)
    }

    let mut builder = ProgramBuilder::new();
    let (true_id, false_id) = branches(&mut builder, depth);
    let mut entry = RegionBuilder::new();
    let predicate = entry.add_input();
    let x = entry.add_input();
    let output = entry.add_instruction(Op::Cond(Cond), vec![predicate, x], 1, vec![true_id, false_id])[0];
    entry.set_outputs(vec![output]);
    let entry_id = builder.seal(entry);
    builder.build(entry_id)
}

/// Builds a one-region program computing `body` over one input.
fn single_region_program(body: impl FnOnce(&mut RegionBuilder<Stored, Op>, usize) -> usize) -> Program<Stored, Op> {
    let mut builder = ProgramBuilder::new();
    let mut region = RegionBuilder::new();
    let x = region.add_input();
    let output = body(&mut region, x);
    region.set_outputs(vec![output]);
    let entry = builder.seal(region);
    builder.build(entry)
}

#[test]
fn test_direct_driver_nested_interpretation_substitutes_captures() {
    // Interpretation with distinct stored-constant (`Stored`) and runtime (`f64`) types, capture substitution
    // through three nesting levels, and the deliberately unbounded interpretation context parameter.
    let program = nested_condition_program(2);
    let context = EagerContext { captures: Rc::new(vec![10.0]) };
    let lift = |constant: &Stored| context.lift(constant);
    assert_eq!(program.interpret_in_context(&context, &lift, vec![1.0, 2.0]), Ok(vec![12.0]));
    assert_eq!(program.interpret_in_context(&context, &lift, vec![0.0, 3.0]), Ok(vec![9.0]));
    // A missing capture surfaces as an interpretation error rather than a silent wrong value.
    let empty = EagerContext { captures: Rc::new(Vec::new()) };
    assert_eq!(
        program.interpret_in_context(&empty, &|constant| empty.lift(constant), vec![1.0, 2.0]),
        Err("missing capture 0".to_string()),
    );
}

#[test]
fn test_attachment_aware_hook_binds_fresh_regional_operations() {
    let true_program = single_region_program(|region, x| {
        let one = region.add_constant(Stored::Literal(1.0));
        region.add_instruction(Op::Prim(Prim::Add), vec![x, one], 1, Vec::new())[0]
    });
    let false_program =
        single_region_program(|region, x| region.add_instruction(Op::Prim(Prim::Mul), vec![x, x], 1, Vec::new())[0]);

    // Eager contexts interpret freshly authored nested computations through the detached access path.
    let context = EagerContext { captures: Rc::new(Vec::new()) };
    assert_eq!(
        context.bind(Op::Cond(Cond), &[], &[], &[1.0, 3.0]),
        Err("operation has no attached regions".to_string()),
    );
    assert_eq!(
        context.bind(Op::Cond(Cond), &[true_program.clone(), false_program.clone()], &[], &[1.0, 3.0]),
        Ok(vec![4.0]),
    );
    assert_eq!(
        context.bind(Op::Cond(Cond), &[true_program.clone(), false_program.clone()], &[], &[0.0, 3.0]),
        Ok(vec![9.0]),
    );
    // Region-free binding uses the same protocol with empty lists.
    assert_eq!(context.bind(Op::Prim(Prim::Add), &[], &[], &[1.0, 2.0]), Ok(vec![3.0]));

    // Staging contexts add the nested computations to the destination program and attach their ids.
    let tracing = TracingContext::new();
    let predicate = tracing.builder.borrow_mut().add_input();
    let x = tracing.builder.borrow_mut().add_input();
    let outputs = tracing.bind(Op::Cond(Cond), &[true_program, false_program], &[], &[predicate, x]).unwrap();
    let staged = tracing.finish(outputs);
    let instruction = &staged.entry_region().instructions[0];
    assert_eq!(instruction.regions.len(), 2);
    // Entry plus two imported attachment regions.
    assert_eq!(staged.regions.len(), 3);
    // The staged program interprets like the eagerly bound one.
    let context = EagerContext { captures: Rc::new(Vec::new()) };
    assert_eq!(
        staged.interpret_in_context(&context, &|constant| context.lift(constant), vec![1.0, 3.0]),
        Ok(vec![4.0]),
    );
}

#[test]
fn test_fresh_stack_transform_drivers_use_direct_bounds() {
    // The load-bearing assertion in this test is that it COMPILES: proving `Op: BatchableOperation<BatchingContext<
    // TracingContext>>`, `Op: DifferentiableOperation<DifferentiationContext<TracingContext>>`, the composed
    // linearize stack, and transposition for a region-attached higher-order variant, with direct per-variant bounds
    // with direct per-variant bounds. Depth is runtime data, so arbitrarily deep nesting adds no obligations.
    let program = nested_condition_program(2);
    let context = EagerContext { captures: Rc::new(vec![10.0]) };
    let lift = |constant: &Stored| context.lift(constant);

    // Batching is a semantic identity in this model, so the batched program interprets identically.
    let batched = program.batched().unwrap();
    assert_eq!(batched.interpret_in_context(&context, &lift, vec![1.0, 2.0]), Ok(vec![12.0]));

    // The fused JVP driver runs the same single `jvp` rule set; its output structure is a stub, so only driver
    // success and output arity are asserted.
    let jvp = program.jvp().unwrap();
    assert_eq!(jvp.entry_region().outputs.len(), 2);

    // The composed linearize stack folds the all-known primal side, so the primal program interprets identically.
    let (primal, residual_count) = program.linearize().unwrap();
    assert_eq!(residual_count, 0);
    assert_eq!(primal.interpret_in_context(&context, &lift, vec![1.0, 2.0]), Ok(vec![12.0]));

    // Transposition is a semantic identity stub; the `&mut` destination context coexisting with a driver that
    // borrows the source is the gate item.
    let transposed = program.transposed().unwrap();
    assert_eq!(transposed.interpret_in_context(&context, &lift, vec![1.0, 2.0]), Ok(vec![12.0]));

    // Partial evaluation against a fixed user context.
    let known = program
        .partially_evaluate_in_context(
            &context,
            &|constant| context.lift(constant),
            vec![PartialValue::Known(1.0), PartialValue::Known(2.0)],
        )
        .unwrap();
    assert_eq!(known.len(), 1);
    assert_eq!(known[0].known(), Some(&12.0));
    let unknown = program
        .partially_evaluate_in_context(
            &context,
            &|constant| context.lift(constant),
            vec![PartialValue::Known(1.0), PartialValue::Unknown],
        )
        .unwrap();
    assert!(unknown[0].known().is_none());
}

#[test]
fn test_callee_interning_and_full_closure_import() {
    let callee = Rc::new(single_region_program(|region, x| {
        let one = region.add_constant(Stored::Literal(1.0));
        region.add_instruction(Op::Prim(Prim::Add), vec![x, one], 1, Vec::new())[0]
    }));
    let equal_but_distinct = Rc::new(single_region_program(|region, x| {
        let one = region.add_constant(Stored::Literal(1.0));
        region.add_instruction(Op::Prim(Prim::Add), vec![x, one], 1, Vec::new())[0]
    }));

    // Interning: one live `Rc` imports once; a structurally equal but distinct program stays distinct.
    let mut builder = ProgramBuilder::<Stored, Op>::new();
    let first = builder.intern_callee(&callee);
    let second = builder.intern_callee(&callee);
    let third = builder.intern_callee(&equal_but_distinct);
    assert_eq!(first, second);
    assert_ne!(first, third);

    // Full-closure import with sharing preserved: a source program whose entry references the same callee root
    // from two instructions imports that callee exactly once.
    let mut source_builder = ProgramBuilder::<Stored, Op>::new();
    let callee_root = source_builder.intern_callee(&callee);
    let mut entry = RegionBuilder::new();
    let x = entry.add_input();
    // The mini-model does not validate operation-declared attachment counts, so a primitive carrying a callee edge
    // stands in for a call operation.
    let first_call = entry.add_instruction(Op::Prim(Prim::Add), vec![x, x], 1, vec![callee_root]);
    let second_call = entry.add_instruction(Op::Prim(Prim::Mul), vec![first_call[0], x], 1, vec![callee_root]);
    entry.set_outputs(vec![second_call[0]]);
    let source_entry = source_builder.seal(entry);
    let source = source_builder.build(source_entry);
    assert_eq!(source.regions.len(), 2);

    let mut destination = ProgramBuilder::<Stored, Op>::new();
    let imported_entry = destination.import_region(&source);
    let imported = destination.build(imported_entry);
    assert_eq!(imported.regions.len(), 2);
    let imported_instructions = &imported.entry_region().instructions;
    assert_eq!(imported_instructions[0].regions, imported_instructions[1].regions);

    // Two lexical imports of one program produce two owned subtrees.
    let mut duplicating = ProgramBuilder::<Stored, Op>::new();
    let first_subtree = duplicating.import_region(&callee);
    let second_subtree = duplicating.import_region(&callee);
    assert_ne!(first_subtree, second_subtree);
}

#[test]
fn test_lazy_origin_resolution_through_region_attachments() {
    // Entry: cond(p, x) with a `true` region producing mul(x, x) and a `false` region yielding a constant (the
    // synthetic-padding shape). The boundary value is the condition result.
    let mut builder = ProgramBuilder::<Stored, Op>::new();
    let mut true_branch = RegionBuilder::new();
    let x = true_branch.add_input();
    let product = true_branch.add_instruction(Op::Prim(Prim::Mul), vec![x, x], 1, Vec::new())[0];
    true_branch.set_outputs(vec![product]);
    let true_id = builder.seal(true_branch);
    let mut false_branch = RegionBuilder::new();
    let _ = false_branch.add_input();
    let zero = false_branch.add_constant(Stored::Literal(0.0));
    false_branch.set_outputs(vec![zero]);
    let false_id = builder.seal(false_branch);
    let mut entry = RegionBuilder::new();
    let predicate = entry.add_input();
    let x = entry.add_input();
    let output = entry.add_instruction(Op::Cond(Cond), vec![predicate, x], 1, vec![true_id, false_id])[0];
    entry.set_outputs(vec![output]);
    let entry_id = builder.seal(entry);
    let program = builder.build(entry_id);

    // Region-provenance resolution: the true branch contributes its mul result; the false branch's constant
    // contributes nothing (padding never surfaces a producer).
    let boundary = ValueId { region: entry_id, atom: output };
    assert_eq!(program.resolve_origins(boundary), Ok(vec![ValueId { region: true_id, atom: product }]));

    // Inputs and constants resolve to empty origins; opaque primitive results resolve to themselves.
    assert_eq!(program.resolve_origins(ValueId { region: entry_id, atom: predicate }), Ok(Vec::new()));
    let mul_result = ValueId { region: true_id, atom: product };
    assert_eq!(program.resolve_origins(mul_result), Ok(vec![mul_result]));

    // First-occurrence deduplication: both condition regions referencing one region resolve that region's producer
    // once. (The mini-model does not enforce unique lexical ownership, which production validation does.)
    let mut shared_builder = ProgramBuilder::<Stored, Op>::new();
    let mut shared = RegionBuilder::new();
    let x = shared.add_input();
    let doubled = shared.add_instruction(Op::Prim(Prim::Add), vec![x, x], 1, Vec::new())[0];
    shared.set_outputs(vec![doubled]);
    let shared_id = shared_builder.seal(shared);
    let mut shared_entry = RegionBuilder::new();
    let predicate = shared_entry.add_input();
    let x = shared_entry.add_input();
    let output = shared_entry.add_instruction(Op::Cond(Cond), vec![predicate, x], 1, vec![shared_id, shared_id])[0];
    shared_entry.set_outputs(vec![output]);
    let shared_entry_id = shared_builder.seal(shared_entry);
    let shared_program = shared_builder.build(shared_entry_id);
    assert_eq!(
        shared_program.resolve_origins(ValueId { region: shared_entry_id, atom: output }),
        Ok(vec![ValueId { region: shared_id, atom: doubled }]),
    );

    // Malformed output provenance errors instead of resolving: a condition instruction with no attached regions.
    let mut malformed_builder = ProgramBuilder::<Stored, Op>::new();
    let mut malformed_entry = RegionBuilder::new();
    let predicate = malformed_entry.add_input();
    let x = malformed_entry.add_input();
    let output = malformed_entry.add_instruction(Op::Cond(Cond), vec![predicate, x], 1, Vec::new())[0];
    malformed_entry.set_outputs(vec![output]);
    let malformed_id = malformed_builder.seal(malformed_entry);
    let malformed = malformed_builder.build(malformed_id);
    assert_eq!(
        malformed.resolve_origins(ValueId { region: malformed_id, atom: output }),
        Err("output provenance references a region index out of range".to_string()),
    );
}

#[test]
fn test_instruction_layout_measurements() {
    // Phase 0 layout measurement: today's instruction shape versus the candidate region-carrying shapes considered
    // during the migration. Production adopted the folded single-vector shape (one region list; edge-kind policies
    // live at the binding boundary); the alternatives remain recorded here for any future layout revisit.
    struct Legacy<O> {
        _operation: O,
        _inputs: Vec<usize>,
        _outputs: Vec<usize>,
    }
    struct Folded<O> {
        _operation: O,
        _inputs: Vec<usize>,
        _outputs: Vec<usize>,
        _regions: Vec<RegionId>,
    }
    struct TwoVectors<O> {
        _operation: O,
        _inputs: Vec<usize>,
        _outputs: Vec<usize>,
        _regions: Vec<RegionId>,
        _callees: Vec<RegionId>,
    }
    let vector = size_of::<Vec<usize>>();
    let legacy = size_of::<Legacy<Op>>();
    let folded = size_of::<Folded<Op>>();
    let two_vectors = size_of::<TwoVectors<Op>>();
    println!("legacy: {legacy} bytes, folded: {folded} bytes, two-vector: {two_vectors} bytes");
    assert_eq!(folded, legacy + vector);
    assert_eq!(two_vectors, legacy + 2 * vector);
}
