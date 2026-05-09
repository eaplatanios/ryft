pub mod differentiation;
pub mod engines;
pub mod errors;
pub mod programs;

pub use differentiation::{LinearOperation, TranspositionContext};
pub use engines::{Engine, ScalarEngine, Tracer, TracerState, TracingContext, TracingEngine};
pub use errors::TracingError;
pub use programs::{Atom, AtomId, Instruction, Program, ProgramBuilder, Traceable, Value};
