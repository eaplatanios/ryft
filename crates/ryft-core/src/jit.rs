// JITing would roughly look like this:
//
// - Tracers that build up symbolic program representations (like `Jaxpr`s).
// - An op that evaluates a `Jaxpr` using XLA or using normal evaluation.
//   This op will need to support things like differentiation to allow differentiation of JITed computations.
