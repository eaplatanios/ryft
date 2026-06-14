use std::fmt::{Debug, Display};

use crate::macros::{check_count, check_types};
use crate::operations::constants::Zero;
use crate::operations::manipulation::{Reshape, Slice, UpdateSlice};
use crate::operations::{InterpretableOperation, Operation, OperationFormatter};
use crate::parameters::Parameterized;
use crate::programs::{Program, ProgramError, Value};
use crate::types::{ArrayType, Shape, Size, Type, TypeError};

// TODO(eaplatanios): Review from here onwards.

/// Canonical operation name for [`ScanOperation`].
pub const SCAN_OPERATION_NAME: &'static str = "scan";

/// [`Operation`] that applies a nested body [`Program`] a static number of times over a loop-carried state while
/// consuming one slice of each stacked input per iteration and stacking the body's per-iteration outputs. This is the
/// statically shaped loop primitive analogous to JAX's
/// [`lax.scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html): where [`WhileOperation`] iterates a
/// data-dependent number of times and therefore supports reverse-mode differentiation only by unrolling in eager
/// domains, `scan` has a static trip count, so its linearization can *store* per-iteration residuals as statically
/// shaped stacks and its linear form transposes totally (see the `tracing_v2` scan rules).
///
/// The body [`Program`] maps `[carry..., x_slice...]` to `[carry..., y_slice...]`: the first
/// [`carry_count`](Self::carry_count) inputs and outputs are the loop-carried state (with identical type signatures),
/// the remaining inputs consume one slice of each stacked input per iteration, and the remaining outputs produce one
/// slice of each stacked output per iteration. The operation's inputs are `[carry..., stacked_xs...]` and its outputs
/// are `[final_carry..., stacked_ys...]`, where each stacked type prepends a static
/// [`length`](Self::length) dimension to the corresponding body slice type. Iteration `i` consumes slice `i` of every
/// stacked input and produces slice `i` of every stacked output; when [`reverse`](Self::reverse) is `true` the
/// iterations visit the slices from `length - 1` down to `0`, but slice `i` of every stacked output still corresponds
/// to slice `i` of the stacked inputs (the visit *order* reverses, the slice *pairing* does not). Transposition of
/// the linear form simply flips [`reverse`](Self::reverse), so no array-reversal operation is ever needed.
///
/// The `length` is stored explicitly so that scans without stacked inputs (pure carry loops with stacked outputs)
/// remain well-defined. All body slice types must be fully static because stacking prepends a static `length`
/// dimension and per-iteration slice extraction must be provably in bounds.
///
/// The optional [`unroll`](Self::unroll) factor (attached via [`with_unroll`](Self::with_unroll)) is a
/// **lowering-only** attribute: interpretation and every transform rule (differentiation, transposition, batching)
/// ignore it semantically but preserve it on whatever scan they re-stage, while lowerings emit `unroll` body copies
/// per loop trip — and a fully unrolled straight-line lowering with no loop at all when `unroll` equals `length`.
///
/// [`WhileOperation`]: crate::operations::control_flow::WhileOperation
#[derive(Clone, Debug)]
pub struct ScanOperation<V, O, T>
where
    T: PartialEq + Type,
    V: Value<T>,
{
    /// Body [`Program`] of this [`ScanOperation`] that maps `[carry..., x_slice...]` to `[carry..., y_slice...]`.
    pub(crate) body: Program<T, V, O, Vec<V>, Vec<V>>,

    /// Number of loop-carried state leaves at the front of the body's inputs and outputs.
    pub(crate) carry_count: usize,

    /// Static trip count of this [`ScanOperation`].
    pub(crate) length: usize,

    /// Boolean indicating whether iterations visit the stacked slices in reverse order.
    pub(crate) reverse: bool,

    /// Lowering-only unroll factor: the number of body copies emitted per loop trip (`1` keeps one body per trip).
    pub(crate) unroll: usize,
}

/// Validates that every dimension of `r#type` is static, reporting a precise error that names the scan `role` (for
/// example, `input 1` or `output 0`) when one is not. Scan body slice types must be fully static because stacking
/// prepends a static length dimension and per-iteration slice extraction must be provably in bounds.
fn check_static_scan_type(role: &str, index: usize, r#type: &ArrayType) -> Result<(), TypeError> {
    for (axis, dimension) in r#type.shape().dimensions().iter().enumerate() {
        if dimension.value().is_none() {
            return Err(TypeError {
                message: format!(
                    "scan body {role} {index} must have a fully static type but axis {axis} of {type} has size \
                     {dimension}",
                    r#type = r#type,
                ),
            });
        }
    }
    Ok(())
}

/// Validates a scan unroll factor against the scan's static trip count: the factor must be at least `1` and must
/// evenly divide `length` (remainder handling is an explicit non-goal). This backs both
/// [`ScanOperation::with_unroll`] and the enum-payload validation of the linear scan variant of
/// [`LinearArrayOperation`](crate::tracing_v2::LinearArrayOperation), whose fields are public and therefore cannot
/// rely on builder-time validation alone.
pub(crate) fn validate_scan_unroll(unroll: usize, length: usize) -> Result<(), TypeError> {
    if unroll == 0 {
        return Err(TypeError { message: "scan unroll factor must be at least 1".to_string() });
    }
    if length % unroll != 0 {
        return Err(TypeError {
            message: format!("scan unroll factor {unroll} must evenly divide the scan length {length}"),
        });
    }
    Ok(())
}

/// Returns the stacked variant of a scan body slice type, prepending a static `length` dimension to its shape.
pub(crate) fn stacked_scan_type(slice_type: &ArrayType, length: usize) -> ArrayType {
    let mut dimensions = Vec::with_capacity(slice_type.rank() + 1);
    dimensions.push(Size::Static(length));
    dimensions.extend(slice_type.shape().dimensions().iter().cloned());
    ArrayType::new(slice_type.data_type(), Shape::new(dimensions))
}

/// Validates `[carry..., stacked_xs...]` input types against a scan body signature and returns the
/// `[carry..., stacked_ys...]` output types. This backs type inference for both [`ScanOperation`] and the linear
/// scan variant of [`LinearArrayOperation`](crate::tracing_v2::LinearArrayOperation).
pub(crate) fn scan_output_types(
    body_input_types: &[ArrayType],
    body_output_types: &[ArrayType],
    carry_count: usize,
    length: usize,
    input_types: &[ArrayType],
) -> Result<Vec<ArrayType>, TypeError> {
    let mut expected_input_types = body_input_types[..carry_count].to_vec();
    expected_input_types
        .extend(body_input_types[carry_count..].iter().map(|slice_type| stacked_scan_type(slice_type, length)));
    check_count!("input", input_types, expected_input_types.len(), TypeError);
    check_types!("scan input", &expected_input_types, input_types);
    let mut output_types = body_output_types[..carry_count].to_vec();
    output_types
        .extend(body_output_types[carry_count..].iter().map(|slice_type| stacked_scan_type(slice_type, length)));
    Ok(output_types)
}

impl<V: Value<ArrayType>, O: Operation<ArrayType>> ScanOperation<V, O, ArrayType> {
    /// Creates a new [`ScanOperation`] with the provided body program, carry count, and static trip count, visiting
    /// the stacked slices in increasing order (use [`Self::with_reverse`] to flip the visit order).
    ///
    /// # Parameters
    ///
    ///   - `body`: Body [`Program`] that maps `[carry..., x_slice...]` to `[carry..., y_slice...]`. Its first
    ///     `carry_count` input and output types must agree, and every input and output type must be fully static.
    ///   - `carry_count`: Number of loop-carried state leaves at the front of the body's inputs and outputs.
    ///   - `length`: Static trip count; each stacked input and output type prepends a static `length` dimension to
    ///     the corresponding body slice type.
    pub fn new(
        body: Program<ArrayType, V, O, Vec<V>, Vec<V>>,
        carry_count: usize,
        length: usize,
    ) -> Result<Self, TypeError> {
        let input_types = body.input_types();
        let output_types = body.output_types();
        if carry_count > input_types.len() {
            return Err(TypeError {
                message: format!("scan carry count {carry_count} exceeds the body input count {}", input_types.len(),),
            });
        }
        if carry_count > output_types.len() {
            return Err(TypeError {
                message: format!("scan carry count {carry_count} exceeds the body output count {}", output_types.len(),),
            });
        }
        check_types!("scan body carry", &input_types[..carry_count], &output_types[..carry_count]);
        for (index, input_type) in input_types.iter().enumerate() {
            check_static_scan_type("input", index, input_type)?;
        }
        for (index, output_type) in output_types.iter().enumerate() {
            check_static_scan_type("output", index, output_type)?;
        }
        Ok(Self { body, carry_count, length, reverse: false, unroll: 1 })
    }

    /// Returns this [`ScanOperation`] with the slice visit order set to `reverse`. When `reverse` is `true`,
    /// iterations visit the stacked slices from `length - 1` down to `0` while slice `i` of every stacked output
    /// still corresponds to slice `i` of the stacked inputs.
    #[inline]
    pub fn with_reverse(mut self, reverse: bool) -> Self {
        self.reverse = reverse;
        self
    }

    /// Returns this [`ScanOperation`] with the lowering unroll factor set to `unroll`. The factor must be at least
    /// `1` and must evenly divide [`length`](Self::length) (remainder handling is an explicit non-goal). Unrolling
    /// is lowering-only: interpretation and transform rules ignore the factor but preserve it, while lowerings emit
    /// `unroll` body copies per loop trip — and a fully unrolled straight-line lowering with no loop at all when
    /// `unroll` equals [`length`](Self::length).
    pub fn with_unroll(mut self, unroll: usize) -> Result<Self, ProgramError> {
        validate_scan_unroll(unroll, self.length)?;
        self.unroll = unroll;
        Ok(self)
    }

    /// Returns the `[carry..., stacked_xs...]` input types of this [`ScanOperation`].
    pub fn input_types(&self) -> Vec<ArrayType> {
        let body_input_types = self.body.input_types();
        let mut input_types = body_input_types[..self.carry_count].to_vec();
        input_types.extend(
            body_input_types[self.carry_count..]
                .iter()
                .map(|slice_type| stacked_scan_type(slice_type, self.length)),
        );
        input_types
    }

    /// Returns the `[final_carry..., stacked_ys...]` output types of this [`ScanOperation`].
    pub fn output_types(&self) -> Vec<ArrayType> {
        let body_output_types = self.body.output_types();
        let mut output_types = body_output_types[..self.carry_count].to_vec();
        output_types.extend(
            body_output_types[self.carry_count..]
                .iter()
                .map(|slice_type| stacked_scan_type(slice_type, self.length)),
        );
        output_types
    }
}

impl<T: PartialEq + Type, V: Value<T>, O: Operation<T>> ScanOperation<V, O, T> {
    /// Returns the body [`Program`] of this [`ScanOperation`] that computes one iteration.
    #[inline]
    pub fn body(&self) -> &Program<T, V, O, Vec<V>, Vec<V>> {
        &self.body
    }

    /// Returns the number of loop-carried state leaves of this [`ScanOperation`].
    #[inline]
    pub fn carry_count(&self) -> usize {
        self.carry_count
    }

    /// Returns the static trip count of this [`ScanOperation`].
    #[inline]
    pub fn length(&self) -> usize {
        self.length
    }

    /// Returns `true` when iterations of this [`ScanOperation`] visit the stacked slices in reverse order.
    #[inline]
    pub fn reverse(&self) -> bool {
        self.reverse
    }

    /// Returns the lowering-only unroll factor of this [`ScanOperation`] (the number of body copies emitted per
    /// loop trip; `1` when no unrolling was requested via [`with_unroll`](Self::with_unroll)).
    #[inline]
    pub fn unroll(&self) -> usize {
        self.unroll
    }
}

impl<T: PartialEq + Type, V: Value<T>, O> Display for ScanOperation<V, O, T>
where
    Self: Operation<T>,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<V: Value<ArrayType>, O: Operation<ArrayType>> Operation<ArrayType> for ScanOperation<V, O, ArrayType> {
    #[inline]
    fn name(&self) -> &'static str {
        SCAN_OPERATION_NAME
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        scan_output_types(
            self.body.input_types().as_slice(),
            self.body.output_types().as_slice(),
            self.carry_count,
            self.length,
            input_types,
        )
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, SCAN_OPERATION_NAME)?.bracketed(|operation| {
            operation.field("carry_count", self.carry_count)?;
            operation.field("length", self.length)?;
            operation.field("reverse", self.reverse)?;
            if self.unroll > 1 {
                operation.field("unroll", self.unroll)?;
            }
            operation.program("body", &self.body)
        })
    }
}

/// Extracts slice `lane` of a stacked value along its leading axis and drops that axis.
///
/// The slice bounds and the squeezed shape are derived from the stacked value's own type, which must be fully static
/// with a leading axis of extent greater than `lane` (guaranteed for stacked scan values by construction).
pub(crate) fn read_scan_lane<V>(stack: &V, lane: usize) -> Result<V, ProgramError>
where
    V: Value<ArrayType> + Slice<Output = V> + Reshape,
{
    let stack_type = stack.r#type().into_owned();
    let dimensions = stack_type
        .shape()
        .dimensions()
        .iter()
        .map(|dimension| {
            dimension.value().ok_or_else(|| {
                TypeError {
                    message: format!("scan lane extraction requires a static stacked type but got {stack_type}"),
                }
                .into()
            })
        })
        .collect::<Result<Vec<usize>, ProgramError>>()?;
    let mut start_indices = vec![0; dimensions.len()];
    start_indices[0] = lane;
    let mut limit_indices = dimensions.clone();
    limit_indices[0] = lane + 1;
    let unit_strides = vec![1; dimensions.len()];
    let lane_value =
        stack.clone().slice(start_indices.as_slice(), limit_indices.as_slice(), unit_strides.as_slice())?;
    lane_value.reshape(Shape::new(dimensions[1..].iter().map(|&dimension| Size::Static(dimension)).collect()))
}

/// Writes `value` as slice `lane` of `accumulator` along its leading axis, prepending a unit axis to `value` first.
fn write_scan_lane<V>(accumulator: V, lane: usize, value: V) -> Result<V, ProgramError>
where
    V: Value<ArrayType> + UpdateSlice<Output = V> + Reshape,
{
    let value_type = value.r#type().into_owned();
    let mut dimensions = Vec::with_capacity(value_type.rank() + 1);
    dimensions.push(Size::Static(1));
    dimensions.extend(value_type.shape().dimensions().iter().cloned());
    let expanded = value.reshape(Shape::new(dimensions))?;
    let mut start_indices = vec![0; value_type.rank() + 1];
    start_indices[0] = lane;
    accumulator.update_slice(expanded, start_indices.as_slice())
}

/// Drives one scan loop over `[carry..., stacked_xs...]` inputs, delegating each iteration's body evaluation to
/// `interpret_lane` and allocating stacked output accumulators through `allocate_zero`.
///
/// This is the single source of truth for scan lane arithmetic: iteration `lane` consumes slice `lane` of every
/// stacked input and writes slice `lane` of every stacked output, visiting lanes from `length - 1` down to `0` when
/// `reverse` is `true` (the visit order reverses while the slice pairing does not). [`ScanOperation`]'s
/// interpretation evaluates the body program directly, while the linear scan interpretation arms instantiate the
/// body's scan-local residual references against each lane's residual values before evaluating it; both share this
/// loop.
///
/// # Parameters
///
///   - `carry_count`: Number of loop-carried state leaves at the front of `inputs`.
///   - `length`: Static trip count.
///   - `reverse`: Whether lanes are visited in reverse order.
///   - `y_slice_types`: Per-iteration stacked output slice types used to allocate the output accumulators.
///   - `inputs`: Flat `[carry..., stacked_xs...]` input values.
///   - `allocate_zero`: Allocates a zero value of the provided stacked output type.
///   - `interpret_lane`: Evaluates one iteration, mapping `(lane, [carry..., x_slice...])` to `[carry...,
///     y_slice...]`.
pub(crate) fn interpret_scan_lanes<V, AllocateZeroFn, InterpretLaneFn>(
    carry_count: usize,
    length: usize,
    reverse: bool,
    y_slice_types: &[ArrayType],
    inputs: &[V],
    mut allocate_zero: AllocateZeroFn,
    mut interpret_lane: InterpretLaneFn,
) -> Result<Vec<V>, ProgramError>
where
    V: Value<ArrayType> + Slice<Output = V> + UpdateSlice<Output = V> + Reshape,
    AllocateZeroFn: FnMut(&ArrayType) -> Result<V, ProgramError>,
    InterpretLaneFn: FnMut(usize, Vec<V>) -> Result<Vec<V>, ProgramError>,
{
    let (carries, stacks) = inputs.split_at(carry_count);
    let mut carries = carries.to_vec();
    let mut accumulators = y_slice_types
        .iter()
        .map(|slice_type| allocate_zero(&stacked_scan_type(slice_type, length)))
        .collect::<Result<Vec<_>, _>>()?;
    let mut lanes: Vec<usize> = (0..length).collect();
    if reverse {
        lanes.reverse();
    }
    for lane in lanes {
        let mut lane_inputs = carries.clone();
        for stack in stacks {
            lane_inputs.push(read_scan_lane(stack, lane)?);
        }
        let mut lane_outputs = interpret_lane(lane, lane_inputs)?;
        check_count!("output", lane_outputs, carry_count + y_slice_types.len(), ProgramError);
        let lane_ys = lane_outputs.split_off(carry_count);
        carries = lane_outputs;
        for (accumulator, lane_y) in accumulators.iter_mut().zip(lane_ys.into_iter()) {
            *accumulator = write_scan_lane(accumulator.clone(), lane, lane_y)?;
        }
    }
    carries.extend(accumulators);
    Ok(carries)
}

impl<V, O> InterpretableOperation<ArrayType, V> for ScanOperation<V, O, ArrayType>
where
    V: Value<ArrayType> + Zero<ArrayType> + Slice<Output = V> + UpdateSlice<Output = V> + Reshape,
    O: InterpretableOperation<ArrayType, V>,
    Vec<V>: Parameterized<V, ParameterStructure: Debug + PartialEq>,
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        self.infer_output_types(input_types.as_slice())?;
        let y_slice_types = self.body.output_types().split_off(self.carry_count);
        interpret_scan_lanes(
            self.carry_count,
            self.length,
            self.reverse,
            y_slice_types.as_slice(),
            inputs,
            |stacked_type| V::zero(stacked_type),
            |_, lane_inputs| self.body.interpret(lane_inputs),
        )
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::operations::compare::ComparisonDirection;
    use crate::parameters::Placeholder;
    use crate::programs::ProgramBuilder;
    use crate::tests::TestArray;
    use crate::tracing_v2::ArrayOperation;
    use crate::types::DataType;

    use super::*;

    /// Test [`Operation`] type used for the nested body programs.
    type TestOperation = ArrayOperation<TestArray, ArrayType>;

    /// Builds a cumulative-product body program that maps `[carry, x]` to `[carry * x, carry * x]`: the new carry is
    /// the running product and each iteration also emits that product as a stacked output slice.
    fn product_body() -> Program<ArrayType, TestArray, TestOperation, Vec<TestArray>, Vec<TestArray>> {
        let mut builder = ProgramBuilder::<ArrayType, TestArray, TestOperation>::new();
        let carry = builder.add_input(ArrayType::scalar(DataType::F64));
        let x = builder.add_input(ArrayType::scalar(DataType::F64));
        let product = builder.add_instruction(TestOperation::Mul, vec![carry, x]).unwrap()[0];
        builder
            .build(vec![product, product], vec![Placeholder, Placeholder], vec![Placeholder, Placeholder])
            .unwrap()
    }

    /// Builds a carry-only body program that maps `[carry]` to `[carry + carry]` with no stacked inputs or outputs.
    fn doubling_body() -> Program<ArrayType, TestArray, TestOperation, Vec<TestArray>, Vec<TestArray>> {
        let mut builder = ProgramBuilder::<ArrayType, TestArray, TestOperation>::new();
        let carry = builder.add_input(ArrayType::scalar(DataType::F64));
        let doubled = builder.add_instruction(TestOperation::Add, vec![carry, carry]).unwrap()[0];
        builder.build(vec![doubled], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    #[test]
    fn test_scan() {
        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let stacked_f64 = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]));
        let operation = ScanOperation::new(product_body(), 1, 3).unwrap();

        // Operation identity and accessors.
        assert_eq!(operation.name(), SCAN_OPERATION_NAME);
        assert_eq!(operation.carry_count(), 1);
        assert_eq!(operation.length(), 3);
        assert!(!operation.reverse());
        assert_eq!(operation.unroll(), 1);
        assert!(operation.with_reverse(true).reverse());
        let operation = ScanOperation::new(product_body(), 1, 3).unwrap();
        assert_eq!(operation.body().input_types(), vec![scalar_f64.clone(), scalar_f64.clone()]);
        assert_eq!(operation.input_types(), vec![scalar_f64.clone(), stacked_f64.clone()]);
        assert_eq!(operation.output_types(), vec![scalar_f64.clone(), stacked_f64.clone()]);
        assert_eq!(
            format!("{operation}"),
            indoc! {"
                scan [
                    carry_count=1,
                    length=3,
                    reverse=false,
                    body={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = mul %0 %1
                        in (%2, %2)
                    },
                ]
            "}
            .trim_end(),
        );

        // Type inference validates the carry and stacked input types and returns the stacked output types.
        assert_eq!(
            operation.infer_output_types(&[scalar_f64.clone(), stacked_f64.clone()]),
            Ok(vec![scalar_f64.clone(), stacked_f64.clone()]),
        );
        assert_eq!(
            operation.infer_output_types(std::slice::from_ref(&scalar_f64)),
            Err(TypeError { message: "expected 2 inputs but got 1".to_string() }),
        );
        assert_eq!(
            operation.infer_output_types(&[scalar_f64.clone(), scalar_f64.clone()]),
            Err(TypeError {
                message: "scan input type signature mismatch: expected [f64[], f64[3]] but got [f64[], f64[]]"
                    .to_string(),
            }),
        );

        // The lowering-only unroll factor must be at least 1 and must evenly divide the scan length; valid factors
        // render only when greater than 1 and interpretation ignores them entirely.
        assert_eq!(
            ScanOperation::new(product_body(), 1, 3).unwrap().with_unroll(0).map(|_| ()),
            Err(ProgramError::Type(TypeError { message: "scan unroll factor must be at least 1".to_string() })),
        );
        assert_eq!(
            ScanOperation::new(product_body(), 1, 3).unwrap().with_unroll(2).map(|_| ()),
            Err(ProgramError::Type(TypeError {
                message: "scan unroll factor 2 must evenly divide the scan length 3".to_string(),
            })),
        );
        let unrolled = ScanOperation::new(product_body(), 1, 3).unwrap().with_unroll(3).unwrap();
        assert_eq!(unrolled.unroll(), 3);
        assert_eq!(
            format!("{unrolled}"),
            indoc! {"
                scan [
                    carry_count=1,
                    length=3,
                    reverse=false,
                    unroll=3,
                    body={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = mul %0 %1
                        in (%2, %2)
                    },
                ]
            "}
            .trim_end(),
        );
        let outputs = unrolled.interpret(&[TestArray::scalar(1.0), TestArray::vector(vec![2.0, 3.0, 4.0])]).unwrap();
        assert_eq!(outputs[0].values, vec![24.0]);
        assert_eq!(outputs[1].values, vec![2.0, 6.0, 24.0]);

        // Construction rejects carry counts that exceed the body signature, mismatched carry types, and dynamically
        // sized body slice types.
        assert_eq!(
            ScanOperation::new(product_body(), 3, 3).map(|_| ()),
            Err(TypeError { message: "scan carry count 3 exceeds the body input count 2".to_string() }),
        );
        let mut builder = ProgramBuilder::<ArrayType, TestArray, TestOperation>::new();
        let carry = builder.add_input(scalar_f64.clone());
        let x = builder.add_input(scalar_f64.clone());
        let product = builder.add_instruction(TestOperation::Mul, vec![carry, x]).unwrap()[0];
        let no_output_body = builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![product], vec![Placeholder, Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            ScanOperation::new(no_output_body, 2, 3).map(|_| ()),
            Err(TypeError { message: "scan carry count 2 exceeds the body output count 1".to_string() }),
        );
        let mut builder = ProgramBuilder::<ArrayType, TestArray, TestOperation>::new();
        let mismatched_carry = builder.add_input(scalar_f64.clone());
        let mismatched_output = builder.add_instruction(TestOperation::ZeroLike, vec![mismatched_carry]).unwrap()[0];
        let mismatched_output = builder
            .add_instruction(
                TestOperation::Compare { direction: ComparisonDirection::Equal },
                vec![mismatched_output, mismatched_carry],
            )
            .unwrap()[0];
        let mismatched_body = builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![mismatched_output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            ScanOperation::new(mismatched_body, 1, 3).map(|_| ()),
            Err(TypeError {
                message: "scan body carry type signature mismatch: expected [f64[]] but got [bool[]]".to_string(),
            }),
        );
        let mut builder = ProgramBuilder::<ArrayType, TestArray, TestOperation>::new();
        let dynamic_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Dynamic(None)]));
        let dynamic_carry = builder.add_input(dynamic_type);
        let dynamic_body = builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![dynamic_carry], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            ScanOperation::new(dynamic_body, 1, 3).map(|_| ()),
            Err(TypeError {
                message: "scan body input 0 must have a fully static type but axis 0 of f64[*] has size *".to_string(),
            }),
        );

        // Interpretation threads the carry while stacking the per-iteration outputs: a cumulative product over
        // `xs = [2, 3, 4]` starting at `1` produces the final carry `24` and the running products `[2, 6, 24]`.
        let outputs = operation.interpret(&[TestArray::scalar(1.0), TestArray::vector(vec![2.0, 3.0, 4.0])]).unwrap();
        assert_eq!(outputs[0].values, vec![24.0]);
        assert_eq!(outputs[1].values, vec![2.0, 6.0, 24.0]);

        // A reversed scan visits the slices from the back while keeping output slice `i` aligned with input slice
        // `i`: the running products visit `4, 3, 2` and land in lanes `2, 1, 0`.
        let reversed = ScanOperation::new(product_body(), 1, 3).unwrap().with_reverse(true);
        let outputs = reversed.interpret(&[TestArray::scalar(1.0), TestArray::vector(vec![2.0, 3.0, 4.0])]).unwrap();
        assert_eq!(outputs[0].values, vec![24.0]);
        assert_eq!(outputs[1].values, vec![24.0, 12.0, 4.0]);

        // A carry-only scan with no stacked inputs or outputs applies the body `length` times.
        let carry_only = ScanOperation::new(doubling_body(), 1, 3).unwrap();
        let outputs = carry_only.interpret(&[TestArray::scalar(1.0)]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].values, vec![8.0]);

        // A zero-length scan returns the initial carries and empty stacked outputs.
        let empty = ScanOperation::new(product_body(), 1, 0).unwrap();
        let empty_stacked_f64 = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(0)]));
        let outputs = empty.interpret(&[TestArray::scalar(1.0), TestArray::new(empty_stacked_f64, vec![])]).unwrap();
        assert_eq!(outputs[0].values, vec![1.0]);
        assert_eq!(outputs[1].values, Vec::<f64>::new());

        // Invalid inputs report precise interpreter errors.
        assert_eq!(
            operation.interpret(&[TestArray::scalar(1.0)]),
            Err(ProgramError::Type(TypeError { message: "expected 2 inputs but got 1".to_string() })),
        );

        // Program rendering uses the canonical operation name and includes the nested body program.
        let mut builder =
            ProgramBuilder::<ArrayType, TestArray, ScanOperation<TestArray, TestOperation, ArrayType>>::new();
        let program_carry = builder.add_input(scalar_f64);
        let program_xs = builder.add_input(stacked_f64);
        let program_outputs = builder.add_instruction(operation, vec![program_carry, program_xs]).unwrap().to_vec();
        let program = builder
            .build::<Vec<TestArray>, Vec<TestArray>>(
                program_outputs,
                vec![Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[3] .
                let %2:f64[], %3:f64[3] = scan [
                    carry_count=1,
                    length=3,
                    reverse=false,
                    body={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = mul %0 %1
                        in (%2, %2)
                    },
                ] %0 %1
                in (%2, %3)
            "}
            .trim_end(),
        );
    }
}
