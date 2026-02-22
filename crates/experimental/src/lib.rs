/////////////////////////////////////////////////////

// pub trait Size {}

// pub struct DynamicSize {
//     pub size: Option<usize>,
// }

// impl Size for DynamicSize {}

// pub struct StaticSize {
//     pub size: usize,
// }

// impl Size for StaticSize {}

// pub struct Shape<DimensionSize: Size> {
//     pub dimensions: Vec<DimensionSize>,
// }
//
/////////////////////////////////////////////////////
//
// pub trait Primitive {
//     fn name(&self) -> String;
// }
//
// pub struct AddPrimitive;
//
// impl Primitive for AddPrimitive {
//     fn name(&self) -> String {
//         "add".to_string()
//     }
// }
//
// pub struct MultiplyPrimitive;
//
// impl Primitive for MultiplyPrimitive {
//     fn name(&self) -> String {
//         "multiply".to_string()
//     }
// }
//
// pub trait AbstractValue {}
//
// pub struct ShapedArray;
//
// impl AbstractValue for ShapedArray {}
//
// impl Add for ShapedArray {
//     type Output = ShapedArray;
//
//     fn add(self, rhs: ShapedArray) -> Self::Output {
//         todo!()
//     }
// }
//
// impl Mul for ShapedArray {
//     type Output = ShapedArray;
//
//     fn mul(self, rhs: ShapedArray) -> Self::Output {
//         todo!()
//     }
// }
//
// pub trait Tracer<Value: AbstractValue> {
//     // __array_priority__ = 1000
//     fn interpreter_state(&self) -> &InterpreterState;
//     fn abstract_value(&self) -> Value;
//     // fn full_lower(&self)
// }
//
// pub struct InterpreterState {
//     pub level: usize,
//     // trace_type: type['Trace']
//     // global_data: Any | None
// }
//
// pub trait Interpreter<Value: AbstractValue, InterpreterTracer: Tracer<Value>> {
//     fn state(&self) -> &InterpreterState;
//
//     /// Constructs a traver for this interpreter, based on the provided non-tracer value.
//     fn pure(&self, value: Value) -> InterpreterTracer;
//
//     /// Lifts the provided tracer from a lower level to the current level.
//     fn lift<LowerLevelInterpreterTracer: Tracer<Value>>(&self, value: LowerLevelInterpreterTracer)
//         -> InterpreterTracer;
//
//     // fn process_op(&self, op: ???, values: Vec<TracerValue<Value>>, params: ???) -> Self;
// }
//
/////////////////////////////////////////////////////
//
// def full_raise(trace: Trace, val: Any) -> Tracer:
//   if not isinstance(val, Tracer):
//     assert type(val) in jax_types
//     return trace.pure(val)
//   level = trace.main.level
//   if val._trace.main is trace.main:
//     return val
//   elif val._trace.main.level < level:
//     return trace.lift(val)
//   elif val._trace.main.level > level:
//     raise Exception(f"Can't lift level {val._trace.main.level} to {level}.")
//   else:  # val._trace.level == level
//     raise Exception(f"Different traces at same level: {val._trace}, {trace}.")

// def add(x, y): return bind1(add_p, x, y)
// def mul(x, y): return bind1(mul_p, x, y)
// def neg(x): return bind1(neg_p, x)
// def sin(x): return bind1(sin_p, x)
// def cos(x): return bind1(cos_p, x)
// def greater(x, y): return bind1(greater_p, x, y)
// def less(x, y): return bind1(less_p, x, y)
// def transpose(x, perm): return bind1(transpose_p, x, perm=perm)
// def broadcast(x, shape, axes): return bind1(broadcast_p, x, shape=shape, axes=axes)
// def reduce_sum(x, axis=None):
//   if axis is None:
//     axis = tuple(range(np.ndim(x)))
//   if type(axis) is int:
//     axis = (axis,)
//   return bind1(reduce_sum_p, x, axis=axis)
//
// def bind1(prim, *args, **params):
//   out, = bind(prim, *args, **params)
//   return out
//
// class MainTrace(NamedTuple):
//   level: int
//   trace_type: type['Trace']
//   global_data: Any | None
//
// trace_stack: list[MainTrace] = []
// dynamic_trace: MainTrace | None = None  # to be employed in Part 3
//
// @contextmanager
// def new_main(trace_type: type['Trace'], global_data=None):
//   level = len(trace_stack)
//   main = MainTrace(level, trace_type, global_data)
//   trace_stack.append(main)
//
//   try:
//     yield main
//   finally:
//     trace_stack.pop()
//
// class Trace:
//   main: MainTrace
//
//   def __init__(self, main: MainTrace) -> None:
//     self.main = main
//
//   def pure(self, val): assert False  # must override
//   def lift(self, val): assert False  # must override
//
//   def process_primitive(self, primitive, tracers, params):
//     assert False  # must override
//
// class Tracer:
//   def __neg__(self): return self.aval._neg(self)
//   def __add__(self, other): return self.aval._add(self, other)
//   def __radd__(self, other): return self.aval._radd(self, other)
//   def __mul__(self, other): return self.aval._mul(self, other)
//   def __rmul__(self, other): return self.aval._rmul(self, other)
//   def __gt__(self, other): return self.aval._gt(self, other)
//   def __lt__(self, other): return self.aval._lt(self, other)
//   def __bool__(self): return self.aval._bool(self)
//   def __nonzero__(self): return self.aval._nonzero(self)
//
//   def __getattr__(self, name):
//     try:
//       return getattr(self.aval, name)
//     except AttributeError:
//       raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")
//
// def swap(f): return lambda x, y: f(y, x)
//
// class ShapedArray:
//   array_abstraction_level = 1
//   shape: tuple[int, ...]
//   dtype: np.dtype
//
//   def __init__(self, shape, dtype):
//     self.shape = shape
//     self.dtype = dtype
//
//   @property
//   def ndim(self):
//     return len(self.shape)
//
//   _neg = staticmethod(neg)
//   _add = staticmethod(add)
//   _radd = staticmethod(swap(add))
//   _mul = staticmethod(mul)
//   _rmul = staticmethod(swap(mul))
//   _gt = staticmethod(greater)
//   _lt = staticmethod(less)
//
//   @staticmethod
//   def _bool(tracer):
//     raise Exception("ShapedArray can't be unambiguously converted to bool")
//
//   @staticmethod
//   def _nonzero(tracer):
//     raise Exception("ShapedArray can't be unambiguously converted to bool")
//
//   def str_short(self):
//     return f'{self.dtype.name}[{",".join(str(d) for d in self.shape)}]'
//
//   def __hash__(self):
//     return hash((self.shape, self.dtype))
//
//   def __eq__(self, other):
//     return (type(self) is type(other) and
//             self.shape == other.shape and self.dtype == other.dtype)
//
//   def __repr__(self):
//     return f"ShapedArray(shape={self.shape}, dtype={self.dtype})"
//
// class ConcreteArray(ShapedArray):
//   array_abstraction_level = 2
//   val: np.ndarray
//
//   def __init__(self, val):
//     self.val = val
//     self.shape = val.shape
//     self.dtype = val.dtype
//
//   @staticmethod
//   def _bool(tracer):
//     return bool(tracer.aval.val)
//
//   @staticmethod
//   def _nonzero(tracer):
//     return bool(tracer.aval.val)
//
// def get_aval(x):
//   if isinstance(x, Tracer):
//     return x.aval
//   elif type(x) in jax_types:
//     return ConcreteArray(np.asarray(x))
//   else:
//     raise TypeError(x)
//
// jax_types = {bool, int, float,
//              np.bool_, np.int32, np.int64, np.float32, np.float64, np.ndarray}
//
// Notice that we actually have two AbstractValues for arrays, representing different levels of abstraction. A ShapedArray represents the set of all possible arrays with a given shape and dtype. A ConcreteArray represents a singleton set consisting of a single array value.
//
// def bind(prim, *args, **params):
//   top_trace = find_top_trace(args)
//   tracers = [full_raise(top_trace, arg) for arg in args]
//   outs = top_trace.process_primitive(prim, tracers, params)
//   return [full_lower(out) for out in outs]
//
// import operator as op
//
// def find_top_trace(xs) -> Trace:
//   top_main = max((x._trace.main for x in xs if isinstance(x, Tracer)),
//                  default=trace_stack[0], key=op.attrgetter('level'))
//   if dynamic_trace and dynamic_trace.level > top_main.level:
//     top_main = dynamic_trace
//   return top_main.trace_type(top_main)
//
// def full_lower(val: Any):
//   if isinstance(val, Tracer):
//     return val.full_lower()
//   else:
//     return val
//
/////////////////////////////////////////////////////
//
// pub fn add(left: usize, right: usize) -> usize {
//     left + right
// }
//
// #[cfg(test)]
// mod tests {
//     use super::*;
//
//     #[test]
//     fn it_works() {
//         let result = add(2, 2);
//         assert_eq!(result, 4);
//     }
// }
