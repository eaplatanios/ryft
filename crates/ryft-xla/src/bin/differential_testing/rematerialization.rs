//! Emits the rematerialization (i.e., gradient checkpointing) cases of the differential harness, which compare Ryft's
//! [`rematerialize`] with JAX's [`jax.checkpoint`](https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint.html).
//!
//! Every case differentiates one rematerialized function of `x: f32[2, 3]` and `w: f32[3, 2]` (and, for the scan cases,
//! of stacked layers `f32[2, 2, 2]`) whose arithmetic is exact in `f32` (i.e., dot products, products, and sums of
//! small integers), so that both frameworks must produce identical values. Each case records its output, its
//! Jacobian-vector product, and its input gradients, which are computed eagerly on core [`CoreArray`]s, together with
//! the values that reverse mode differentiation saves, which [`saved_residuals`] reports. The saved values are encoded
//! numerically as their count and as their element counts and bit widths, sorted by element count and then by bit
//! width, because the two frameworks report them in different orders and describe their sources differently. The
//! StableHLO module of each case is the XLA lowering of the staged output and gradients, whose recomputation must sit
//! behind an optimization barrier.

use std::collections::BTreeMap;
use std::error::Error;

use ryft_core::{
    Add, Array as CoreArray, ArrayOperation, ArrayType, Broadcast, Compare, ComparisonDirection, Condition, DataType,
    DomainTracer, Dot, DotDimensionNumbers, DotsSavable, DotsWithNoBatchDimensionsSavable, EverythingSavable, Memory,
    Mul, NothingSavable, ProgramError, Reduce, ReductionKind, SaveAndOffloadOnlyTheseNames, SaveAnyNamesButThese,
    SaveFromBothPolicies, SaveOnlyTheseNames, SavedResidual, ScanOptions, StaticShape, StopGradient, Tag, Tracer,
    TracingContext, Typed, ValueProjection, ZeroLike, composite, differentiate_at, rematerialize, saved_residuals,
    scan,
};
use ryft_xla::Domain;
use ryft_xla::experimental::{ArrayTracer, TracedProgram, trace};

use super::{DifferentialObservation, SCHEMA};

/// Tracer over which the bodies of the eagerly evaluated rematerialized functions are traced.
type CoreTracer = Tracer<TracingContext<CoreArray, ArrayOperation<CoreArray>>>;

/// Tracer over which the bodies of the rematerialized functions staged for XLA are traced.
type XlaTracer = DomainTracer<Domain<'static>>;

/// Stable identifiers and emitters of the rematerialization cases, in registry order.
pub(super) const CASES: [(&str, fn() -> Result<DifferentialObservation, Box<dyn Error>>); 14] = [
    ("rematerialization_nothing_saveable", emit_nothing_saveable),
    ("rematerialization_everything_saveable", emit_everything_saveable),
    ("rematerialization_dots_saveable", emit_dots_saveable),
    ("rematerialization_dots_with_no_batch_dims_saveable", emit_dots_with_no_batch_dims_saveable),
    ("rematerialization_save_only_these_names", emit_save_only_these_names),
    ("rematerialization_save_any_names_but_these", emit_save_any_names_but_these),
    ("rematerialization_save_from_both_policies", emit_save_from_both_policies),
    ("rematerialization_save_and_offload_only_these_names", emit_save_and_offload_only_these_names),
    ("rematerialization_dots_saveable_call", emit_dots_saveable_call),
    ("rematerialization_dots_saveable_condition", emit_dots_saveable_condition),
    ("rematerialization_dots_saveable_scan", emit_dots_saveable_scan),
    ("rematerialization_dots_saveable_known_condition", emit_dots_saveable_known_condition),
    ("rematerialization_dots_saveable_known_scan", emit_dots_saveable_known_scan),
    ("rematerialization_nested_policies", emit_nested_policies),
];

/// Returns the dot product dimensions of a matrix multiplication.
fn matrix_multiplication() -> DotDimensionNumbers {
    DotDimensionNumbers::new(vec![1], vec![0], vec![], vec![])
}

/// Returns the dot product dimensions that contract the rows of two matrices pairwise, batching over their rows.
fn row_dot() -> DotDimensionNumbers {
    DotDimensionNumbers::new(vec![1], vec![1], vec![0], vec![0])
}

/// Computes `y = tag(x · w, "dot")`, `z = tag(y ⊙ y, "square")`, and `b = tag(rowwise(z · y), "batched")`, and returns
/// `b ⊙ b`. The body thus contains one dot product without batching dimensions, one batched dot product, and three
/// tags, which the policy cases select differently.
fn policy_body<V: Clone + Dot + Mul + Tag>(x: V, w: V) -> Result<V, ProgramError> {
    let y = x.dot(&w, &matrix_multiplication())?.tag("dot")?;
    let z = y.mul(&y)?.tag("square")?;
    let b = z.dot(&y, &row_dot())?.tag("batched")?;
    b.mul(&b)
}

/// Returns `tag(y ⊙ y, "square")` for `y = tag(x · w, "dot")`, the body of the inner rematerialized function of the
/// nested case.
fn nested_body<V: Clone + Dot + Mul + Tag>(x: V, w: V) -> Result<V, ProgramError> {
    let y = x.dot(&w, &matrix_multiplication())?.tag("dot")?;
    y.mul(&y)?.tag("square")
}

/// Returns `y ⊙ y` for `y = x · w`, the body of the call in the call case.
fn call_body<V: Dot + Mul>(x: V, w: V) -> Result<V, ProgramError> {
    let y = x.dot(&w, &matrix_multiplication())?;
    y.mul(&y)
}

/// Returns `t ⊙ t` for `t = y · y`, the `true` branch of the condition case.
fn condition_true_branch<V: Dot + Mul>(y: V) -> Result<V, ProgramError> {
    let t = y.dot(&y, &matrix_multiplication())?;
    t.mul(&t)
}

/// Returns `y ⊙ y ⊙ y`, the `false` branch of the condition case, which contains no dot product.
fn condition_false_branch<V: Mul>(y: V) -> Result<V, ProgramError> {
    y.mul(&y)?.mul(&y)
}

/// Returns the predicate of the condition case, `sum(x) > 0`.
fn condition_predicate<V: Compare + Reduce + ZeroLike>(x: &V) -> Result<V, ProgramError> {
    let total = x.reduce(&[0, 1], ReductionKind::Sum)?;
    total.compare(&total.zero_like()?, ComparisonDirection::GreaterThan)
}

/// Returns `t ⊙ t` for `t = h · layer`, one iteration of the scan case.
fn scan_step<V: Dot + Mul>(hidden: V, layer: V) -> Result<V, ProgramError> {
    let t = hidden.dot(&layer, &matrix_multiplication())?;
    t.mul(&t)
}

/// Returns `broadcast(c)` of shape `[3, 2, 2]`, which expands the compact value `c: f32[2, 2]` of the cases whose
/// regions carry no gradients.
fn expand<V: Broadcast + Typed<Type = ArrayType>>(c: V) -> Result<V, ProgramError> {
    c.broadcast_to(StaticShape::new(vec![3, 2, 2]))
}

/// Returns `broadcast(wᵀ · w)`, the `true` branch of the known condition case.
fn known_condition_true_branch<V: Broadcast + Dot + Typed<Type = ArrayType>>(w: V) -> Result<V, ProgramError> {
    expand(w.dot(&w, &DotDimensionNumbers::new(vec![0], vec![0], vec![], vec![]))?)
}

/// Returns `e + e` for `e = broadcast(wᵀ · w)`, the `false` branch of the known condition case.
fn known_condition_false_branch<V: Add + Broadcast + Dot + Typed<Type = ArrayType>>(w: V) -> Result<V, ProgramError> {
    let e = known_condition_true_branch(w)?;
    e.add(&e)
}

/// Returns `h + broadcast(layer · layer)`, one iteration of the known scan case.
fn known_scan_step<V: Add + Broadcast + Dot + Typed<Type = ArrayType>>(hidden: V, layer: V) -> Result<V, ProgramError> {
    hidden.add(&expand(layer.dot(&layer, &matrix_multiplication())?)?)
}

/// Returns `(x · w) ⊙ Σₖ eₖ` for the expansion `e` of the cases whose regions carry no gradients.
fn known_region_output<V: Dot + Mul + Reduce>(x: V, w: V, e: V) -> Result<V, ProgramError> {
    x.dot(&w, &matrix_multiplication())?.mul(&e.reduce(&[0], ReductionKind::Sum)?)
}

/// Returns the [`ArrayType`] member of an XLA tracer.
fn project(value: XlaTracer) -> Result<ArrayTracer, ProgramError> {
    Ok(ValueProjection::<ArrayType>::into_projected(value)?)
}

/// Returns the `f32` array of the provided shape and values.
fn array(shape: &[usize], values: &[f32]) -> Result<CoreArray, ProgramError> {
    CoreArray::from_elements::<f32>(ArrayType::new_static(DataType::F32, shape.to_vec()), values)
}

/// Returns the flattened values of `array`, which are exact `f32` values.
fn values(array: CoreArray) -> Vec<Vec<f32>> {
    vec![array.to_f64s().into_iter().map(|value| value as f32).collect()]
}

/// Returns the shared inputs `x` and `w` together with their tangents `ẋ` and `ẇ`.
fn inputs() -> Result<[CoreArray; 4], ProgramError> {
    Ok([
        array(&[2, 3], &[1.0, 2.0, -1.0, 0.0, 1.0, 2.0])?,
        array(&[3, 2], &[1.0, -1.0, 2.0, 0.0, -1.0, 1.0])?,
        array(&[2, 3], &[1.0, 0.0, 0.0, 0.0, -1.0, 1.0])?,
        array(&[3, 2], &[0.0, 1.0, 1.0, 0.0, 0.0, 0.0])?,
    ])
}

/// Records the count of `residuals` together with their element counts and bit widths, sorted by element count and
/// then by bit width.
fn record_residuals(
    observations: &mut BTreeMap<&'static str, Vec<Vec<f32>>>,
    residuals: &[SavedResidual<ArrayType>],
) -> Result<(), Box<dyn Error>> {
    let mut facts = residuals
        .iter()
        .map(|residual| {
            let r#type = residual.r#type();
            let element_count = r#type.element_count()?.ok_or("saved residuals must have static shapes")?;
            Ok((element_count, r#type.data_type().bit_width()))
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    facts.sort();
    observations.insert("residual_count", vec![vec![facts.len() as f32]]);
    observations.insert("residual_element_counts", vec![facts.iter().map(|(count, _)| *count as f32).collect()]);
    observations.insert("residual_bit_widths", vec![facts.iter().map(|(_, width)| *width as f32).collect()]);
    Ok(())
}

/// Inserts into `$observations` the output (named `output`), Jacobian-vector product (named `jvp`), and input gradients
/// (named after the inputs with a `_gradient` suffix) of the rematerialized function `$function` at the core arrays
/// bound to the identifiers `$input`, for the tangents `$tangent` and the output cotangent `$seed`. Every name starts
/// with `$prefix`. The function is called once per transform, because each transform traces a different tracer type.
macro_rules! observe {
    // Evaluates the function under each transform, binding the gradients to the input identifiers.
    (
        $observations:ident,
        $prefix:literal,
        $function:ident,
        ($($input:ident),+),
        ($($tangent:expr),+),
        $seed:expr $(,)?
    ) => {{
        let (output, tangent) = differentiate_at(($($input.clone()),+))
            .jvp(($($tangent),+), |($($input),+)| $function.call(($($input),+)))?;
        let (_, pullback) = differentiate_at(($($input.clone()),+)).vjp(|($($input),+)| $function.call(($($input),+)))?;
        let ($($input),+) = pullback.apply_materialized($seed)?;
        $observations.insert(concat!($prefix, "output"), values(output));
        $observations.insert(concat!($prefix, "jvp"), values(tangent));
        $($observations.insert(concat!($prefix, stringify!($input), "_gradient"), values($input));)+
    }};
}

/// Returns the StableHLO module of the staged output and input gradients of the rematerialized function `$function`
/// of XLA tracers, at the types of the core arrays bound to the identifiers `$input` and the output cotangent `$seed`.
macro_rules! stablehlo {
    // Traces the output and gradients over the input types and returns them as one flat list of arrays.
    ($function:ident, ($($input:ident),+), $seed:ident $(,)?) => {{
        let traced: TracedProgram<_, Vec<ArrayType>> = trace(
            move |($($input),+, $seed)| {
                // The closure inputs are inferred from the input types, so their projection is named explicitly.
                $(let $input = ArrayTracer::into_value($input);)+
                let (output, pullback) =
                    differentiate_at(($($input),+)).vjp(|($($input),+)| $function.call(($($input),+))).unwrap();
                let ($($input),+) = pullback.apply_materialized(ArrayTracer::into_value($seed)).unwrap();
                vec![project(output).unwrap(), $(project($input).unwrap()),+]
            },
            ($($input.r#type().into_owned()),+, $seed.r#type().into_owned()),
        )?;
        traced.to_mlir_module("main")?
    }};
}

/// Emits the observations and StableHLO module of the rematerialized functions `$core` and `$staged` of `(x, w)`,
/// whose bodies are traced over [`CoreTracer`]s and [`XlaTracer`]s, respectively, for the output cotangent `$seed`.
macro_rules! emit_case {
    // Functions whose output is a vector use the cotangent `[1, -1]`.
    ($case_id:literal, $core:expr, $staged:expr $(,)?) => {
        emit_case!($case_id, $core, $staged, array(&[2], &[1.0, -1.0])?)
    };

    // Functions with any other output use an explicit cotangent.
    ($case_id:literal, $core:expr, $staged:expr, $seed:expr $(,)?) => {{
        let core = $core;
        let staged = $staged;
        let [x, w, x_tangent, w_tangent] = inputs()?;
        let seed = $seed;
        let mut observations = BTreeMap::new();
        observe!(observations, "", core, (x, w), (x_tangent, w_tangent), seed.clone());
        let residuals = saved_residuals(
            |(x, w): (CoreTracer, CoreTracer)| core.call((x, w)),
            (x.r#type().into_owned(), w.r#type().into_owned()),
        )?;
        record_residuals(&mut observations, &residuals)?;
        Ok(DifferentialObservation {
            schema: SCHEMA,
            case_id: $case_id.into(),
            observations,
            staging: None,
            stablehlo: Some(stablehlo!(staged, (x, w), seed)),
        })
    }};
}

/// Emits [`policy_body`] rematerialized with [`NothingSavable`], which saves only the inputs.
fn emit_nothing_saveable() -> Result<DifferentialObservation, Box<dyn Error>> {
    emit_case!(
        "rematerialization_nothing_saveable",
        rematerialize(|(x, w): (CoreTracer, CoreTracer)| policy_body(x, w)).with_policy(NothingSavable),
        rematerialize(|(x, w): (XlaTracer, XlaTracer)| Ok(policy_body(project(x)?, project(w)?)?.into_value()))
            .with_policy(NothingSavable),
    )
}

/// Emits [`policy_body`] rematerialized with [`EverythingSavable`].
fn emit_everything_saveable() -> Result<DifferentialObservation, Box<dyn Error>> {
    emit_case!(
        "rematerialization_everything_saveable",
        rematerialize(|(x, w): (CoreTracer, CoreTracer)| policy_body(x, w)).with_policy(EverythingSavable),
        rematerialize(|(x, w): (XlaTracer, XlaTracer)| Ok(policy_body(project(x)?, project(w)?)?.into_value()))
            .with_policy(EverythingSavable),
    )
}

/// Emits [`policy_body`] rematerialized with [`DotsSavable`], which saves both dot products.
fn emit_dots_saveable() -> Result<DifferentialObservation, Box<dyn Error>> {
    emit_case!(
        "rematerialization_dots_saveable",
        rematerialize(|(x, w): (CoreTracer, CoreTracer)| policy_body(x, w)).with_policy(DotsSavable),
        rematerialize(|(x, w): (XlaTracer, XlaTracer)| Ok(policy_body(project(x)?, project(w)?)?.into_value()))
            .with_policy(DotsSavable),
    )
}

/// Emits [`policy_body`] rematerialized with [`DotsWithNoBatchDimensionsSavable`], which saves only the dot product
/// without batching dimensions.
fn emit_dots_with_no_batch_dims_saveable() -> Result<DifferentialObservation, Box<dyn Error>> {
    emit_case!(
        "rematerialization_dots_with_no_batch_dims_saveable",
        rematerialize(|(x, w): (CoreTracer, CoreTracer)| policy_body(x, w))
            .with_policy(DotsWithNoBatchDimensionsSavable),
        rematerialize(|(x, w): (XlaTracer, XlaTracer)| Ok(policy_body(project(x)?, project(w)?)?.into_value()))
            .with_policy(DotsWithNoBatchDimensionsSavable),
    )
}

/// Emits [`policy_body`] rematerialized with [`SaveOnlyTheseNames`] selecting the `square` tag.
fn emit_save_only_these_names() -> Result<DifferentialObservation, Box<dyn Error>> {
    emit_case!(
        "rematerialization_save_only_these_names",
        rematerialize(|(x, w): (CoreTracer, CoreTracer)| policy_body(x, w))
            .with_policy(SaveOnlyTheseNames::new(["square"])),
        rematerialize(|(x, w): (XlaTracer, XlaTracer)| Ok(policy_body(project(x)?, project(w)?)?.into_value()))
            .with_policy(SaveOnlyTheseNames::new(["square"])),
    )
}

/// Emits [`policy_body`] rematerialized with [`SaveAnyNamesButThese`] excluding the `square` tag, which saves the
/// `dot` and `batched` tags.
fn emit_save_any_names_but_these() -> Result<DifferentialObservation, Box<dyn Error>> {
    emit_case!(
        "rematerialization_save_any_names_but_these",
        rematerialize(|(x, w): (CoreTracer, CoreTracer)| policy_body(x, w))
            .with_policy(SaveAnyNamesButThese::new(["square"])),
        rematerialize(|(x, w): (XlaTracer, XlaTracer)| Ok(policy_body(project(x)?, project(w)?)?.into_value()))
            .with_policy(SaveAnyNamesButThese::new(["square"])),
    )
}

/// Emits [`policy_body`] rematerialized with the union of [`DotsWithNoBatchDimensionsSavable`] and
/// [`SaveOnlyTheseNames`] selecting the `square` tag.
fn emit_save_from_both_policies() -> Result<DifferentialObservation, Box<dyn Error>> {
    let policy = SaveFromBothPolicies::new(DotsWithNoBatchDimensionsSavable, SaveOnlyTheseNames::new(["square"]));
    emit_case!(
        "rematerialization_save_from_both_policies",
        rematerialize(|(x, w): (CoreTracer, CoreTracer)| policy_body(x, w)).with_policy(policy.clone()),
        rematerialize(|(x, w): (XlaTracer, XlaTracer)| Ok(policy_body(project(x)?, project(w)?)?.into_value()))
            .with_policy(policy),
    )
}

/// Emits [`policy_body`] rematerialized with [`SaveAndOffloadOnlyTheseNames`], which saves the `dot` tag and offloads
/// the `square` tag to pinned host memory.
fn emit_save_and_offload_only_these_names() -> Result<DifferentialObservation, Box<dyn Error>> {
    let policy = SaveAndOffloadOnlyTheseNames::new(["dot"], ["square"], Memory::Host { pinned: true })?;
    emit_case!(
        "rematerialization_save_and_offload_only_these_names",
        rematerialize(|(x, w): (CoreTracer, CoreTracer)| policy_body(x, w)).with_policy(policy.clone()),
        rematerialize(|(x, w): (XlaTracer, XlaTracer)| Ok(policy_body(project(x)?, project(w)?)?.into_value()))
            .with_policy(policy),
    )
}

/// Emits a rematerialized function with [`DotsSavable`] that returns `z ⊙ z` for `z` computed by a nested call of
/// [`call_body`], which is the analogue of a `jax.jit` function called inside `jax.checkpoint`.
fn emit_dots_saveable_call() -> Result<DifferentialObservation, Box<dyn Error>> {
    let core_call = composite(|(x, w): (CoreTracer, CoreTracer)| call_body(x, w));
    let staged_call = composite(|(x, w): (XlaTracer, XlaTracer)| Ok(call_body(project(x)?, project(w)?)?.into_value()));
    emit_case!(
        "rematerialization_dots_saveable_call",
        rematerialize(move |(x, w): (CoreTracer, CoreTracer)| {
            let z = core_call.call((x, w))?;
            z.mul(&z)
        })
        .with_policy(DotsSavable),
        rematerialize(move |(x, w): (XlaTracer, XlaTracer)| {
            let z = project(staged_call.call((x, w))?)?;
            Ok(z.mul(&z)?.into_value())
        })
        .with_policy(DotsSavable),
        array(&[2, 2], &[1.0, -1.0, 2.0, 1.0])?,
    )
}

/// Emits a rematerialized function with [`DotsSavable`] that applies a condition to `y = x · w`, whose `true` branch
/// computes a dot product and whose `false` branch does not. The predicate, `sum(x) > 0`, holds at `x` and fails at
/// `-x`, whose values are recorded under names that start with `false_branch_`.
fn emit_dots_saveable_condition() -> Result<DifferentialObservation, Box<dyn Error>> {
    let core = rematerialize(|(x, w): (CoreTracer, CoreTracer)| {
        let y = x.dot(&w, &matrix_multiplication())?;
        condition_predicate(&x)?.condition(condition_true_branch, condition_false_branch, y)
    })
    .with_policy(DotsSavable);
    let staged = rematerialize(|(x, w): (XlaTracer, XlaTracer)| {
        let x = project(x)?;
        let y = x.dot(&project(w)?, &matrix_multiplication())?;
        condition_predicate(&x)?.into_value().condition(
            |y| Ok(condition_true_branch(ValueProjection::<ArrayType>::into_projected(y)?)?.into_value()),
            |y| Ok(condition_false_branch(ValueProjection::<ArrayType>::into_projected(y)?)?.into_value()),
            y.into_value(),
        )
    })
    .with_policy(DotsSavable);
    let [x, w, x_tangent, w_tangent] = inputs()?;
    let negated = array(&[2, 3], &[-1.0, -2.0, 1.0, 0.0, -1.0, -2.0])?;
    let seed = array(&[2, 2], &[1.0, -1.0, 2.0, 1.0])?;
    let mut observations = BTreeMap::new();
    observe!(observations, "", core, (x, w), (x_tangent.clone(), w_tangent.clone()), seed.clone());
    {
        // The gradients are named after the input identifiers, so the negated input is bound to `x` as well.
        let x = negated;
        observe!(observations, "false_branch_", core, (x, w), (x_tangent, w_tangent), seed.clone());
    }
    let residuals = saved_residuals(
        |(x, w): (CoreTracer, CoreTracer)| core.call((x, w)),
        (x.r#type().into_owned(), w.r#type().into_owned()),
    )?;
    record_residuals(&mut observations, &residuals)?;
    Ok(DifferentialObservation {
        schema: SCHEMA,
        case_id: "rematerialization_dots_saveable_condition".into(),
        observations,
        staging: None,
        stablehlo: Some(stablehlo!(staged, (x, w), seed)),
    })
}

/// Emits a rematerialized function with [`DotsSavable`] that scans [`scan_step`] over two stacked `layers`, starting
/// from `y = x · w`.
fn emit_dots_saveable_scan() -> Result<DifferentialObservation, Box<dyn Error>> {
    let core = rematerialize(|(x, w, layers): (CoreTracer, CoreTracer, CoreTracer)| {
        let y = x.dot(&w, &matrix_multiplication())?;
        let (hidden, ()) = scan(|_, hidden, layer| Ok((scan_step(hidden, layer)?, ())), y, layers, ScanOptions::new())?;
        Ok(hidden)
    })
    .with_policy(DotsSavable);
    let staged = rematerialize(|(x, w, layers): (XlaTracer, XlaTracer, XlaTracer)| {
        let y = project(x)?.dot(&project(w)?, &matrix_multiplication())?;
        let (hidden, ()) = scan(
            |_, hidden, layer| {
                let hidden = ValueProjection::<ArrayType>::into_projected(hidden)?;
                let layer = ValueProjection::<ArrayType>::into_projected(layer)?;
                Ok((scan_step(hidden, layer)?.into_value(), ()))
            },
            y.into_value(),
            layers,
            ScanOptions::new(),
        )?;
        Ok(hidden)
    })
    .with_policy(DotsSavable);
    let [x, w, x_tangent, w_tangent] = inputs()?;
    let layers = array(&[2, 2, 2], &[1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 1.0, 1.0])?;
    let layers_tangent = array(&[2, 2, 2], &[0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0])?;
    let seed = array(&[2, 2], &[1.0, -1.0, 2.0, 1.0])?;
    let mut observations = BTreeMap::new();
    observe!(observations, "", core, (x, w, layers), (x_tangent, w_tangent, layers_tangent), seed.clone());
    let residuals = saved_residuals(
        |(x, w, layers): (CoreTracer, CoreTracer, CoreTracer)| core.call((x, w, layers)),
        (x.r#type().into_owned(), w.r#type().into_owned(), layers.r#type().into_owned()),
    )?;
    record_residuals(&mut observations, &residuals)?;
    Ok(DifferentialObservation {
        schema: SCHEMA,
        case_id: "rematerialization_dots_saveable_scan".into(),
        observations,
        staging: None,
        stablehlo: Some(stablehlo!(staged, (x, w, layers), seed)),
    })
}

/// Emits a rematerialized function with [`DotsSavable`] whose output [`known_region_output`] reads the output of a
/// condition whose inputs carry no gradients. The condition expands the compact `wᵀ · w` of its branch into `f32[3, 2,
/// 2]` (refer to [`known_condition_true_branch`] and [`known_condition_false_branch`]), so the policy saves that
/// compact value rather than the expansion, as JAX does. The predicate, `sum(x) > 0`, holds at `x` and fails at `-x`,
/// whose values are recorded under names that start with `false_branch_`.
fn emit_dots_saveable_known_condition() -> Result<DifferentialObservation, Box<dyn Error>> {
    let core = rematerialize(|(x, w): (CoreTracer, CoreTracer)| {
        let (known_x, known_w) = (x.stop_gradient()?, w.stop_gradient()?);
        let e = condition_predicate(&known_x)?.condition(
            known_condition_true_branch,
            known_condition_false_branch,
            known_w,
        )?;
        known_region_output(x, w, e)
    })
    .with_policy(DotsSavable);
    let staged = rematerialize(|(x, w): (XlaTracer, XlaTracer)| {
        let (x, w) = (project(x)?, project(w)?);
        let (known_x, known_w) = (x.stop_gradient()?, w.stop_gradient()?);
        let e = condition_predicate(&known_x)?.into_value().condition(
            |w| Ok(known_condition_true_branch(ValueProjection::<ArrayType>::into_projected(w)?)?.into_value()),
            |w| Ok(known_condition_false_branch(ValueProjection::<ArrayType>::into_projected(w)?)?.into_value()),
            known_w.into_value(),
        )?;
        Ok(known_region_output(x, w, project(e)?)?.into_value())
    })
    .with_policy(DotsSavable);
    let [x, w, x_tangent, w_tangent] = inputs()?;
    let negated = array(&[2, 3], &[-1.0, -2.0, 1.0, 0.0, -1.0, -2.0])?;
    let seed = array(&[2, 2], &[1.0, -1.0, 2.0, 1.0])?;
    let mut observations = BTreeMap::new();
    observe!(observations, "", core, (x, w), (x_tangent.clone(), w_tangent.clone()), seed.clone());
    {
        // The gradients are named after the input identifiers, so the negated input is bound to `x` as well.
        let x = negated;
        observe!(observations, "false_branch_", core, (x, w), (x_tangent, w_tangent), seed.clone());
    }
    let residuals = saved_residuals(
        |(x, w): (CoreTracer, CoreTracer)| core.call((x, w)),
        (x.r#type().into_owned(), w.r#type().into_owned()),
    )?;
    record_residuals(&mut observations, &residuals)?;
    Ok(DifferentialObservation {
        schema: SCHEMA,
        case_id: "rematerialization_dots_saveable_known_condition".into(),
        observations,
        staging: None,
        stablehlo: Some(stablehlo!(staged, (x, w), seed)),
    })
}

/// Emits a rematerialized function with [`DotsSavable`] whose output [`known_region_output`] reads the final carry of a
/// scan whose inputs carry no gradients. The scan adds the expansion of the compact `layer · layer` into
/// `f32[3, 2, 2]` to its carry on each iteration (refer to [`known_scan_step`]), starting from the expansion of
/// `Σ layers`, so the policy saves the stacked compact values and the layers from which the initial carry is
/// recomputed, rather than the final carry, as JAX does.
fn emit_dots_saveable_known_scan() -> Result<DifferentialObservation, Box<dyn Error>> {
    let core = rematerialize(|(x, w, layers): (CoreTracer, CoreTracer, CoreTracer)| {
        let layers = layers.stop_gradient()?;
        let initial = expand(layers.reduce(&[0], ReductionKind::Sum)?)?;
        let (e, ()) =
            scan(|_, hidden, layer| Ok((known_scan_step(hidden, layer)?, ())), initial, layers, ScanOptions::new())?;
        known_region_output(x, w, e)
    })
    .with_policy(DotsSavable);
    let staged = rematerialize(|(x, w, layers): (XlaTracer, XlaTracer, XlaTracer)| {
        let layers = project(layers)?.stop_gradient()?;
        let initial = expand(layers.reduce(&[0], ReductionKind::Sum)?)?;
        let (e, ()) = scan(
            |_, hidden, layer| {
                let hidden = ValueProjection::<ArrayType>::into_projected(hidden)?;
                let layer = ValueProjection::<ArrayType>::into_projected(layer)?;
                Ok((known_scan_step(hidden, layer)?.into_value(), ()))
            },
            initial.into_value(),
            layers.into_value(),
            ScanOptions::new(),
        )?;
        Ok(known_region_output(project(x)?, project(w)?, project(e)?)?.into_value())
    })
    .with_policy(DotsSavable);
    let [x, w, x_tangent, w_tangent] = inputs()?;
    let layers = array(&[2, 2, 2], &[1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 1.0, 1.0])?;
    let layers_tangent = array(&[2, 2, 2], &[0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0])?;
    let seed = array(&[2, 2], &[1.0, -1.0, 2.0, 1.0])?;
    let mut observations = BTreeMap::new();
    observe!(observations, "", core, (x, w, layers), (x_tangent, w_tangent, layers_tangent), seed.clone());
    let residuals = saved_residuals(
        |(x, w, layers): (CoreTracer, CoreTracer, CoreTracer)| core.call((x, w, layers)),
        (x.r#type().into_owned(), w.r#type().into_owned(), layers.r#type().into_owned()),
    )?;
    record_residuals(&mut observations, &residuals)?;
    Ok(DifferentialObservation {
        schema: SCHEMA,
        case_id: "rematerialization_dots_saveable_known_scan".into(),
        observations,
        staging: None,
        stablehlo: Some(stablehlo!(staged, (x, w, layers), seed)),
    })
}

/// Emits a rematerialized function with [`SaveOnlyTheseNames`] selecting the `square` tag that returns `v ⊙ v` for the
/// output `v` of [`nested_body`] rematerialized with [`DotsSavable`].
fn emit_nested_policies() -> Result<DifferentialObservation, Box<dyn Error>> {
    let core_inner = rematerialize(|(x, w): (CoreTracer, CoreTracer)| nested_body(x, w)).with_policy(DotsSavable);
    let staged_inner =
        rematerialize(|(x, w): (XlaTracer, XlaTracer)| Ok(nested_body(project(x)?, project(w)?)?.into_value()))
            .with_policy(DotsSavable);
    emit_case!(
        "rematerialization_nested_policies",
        rematerialize(move |(x, w): (CoreTracer, CoreTracer)| {
            let v = core_inner.call((x, w))?;
            v.mul(&v)
        })
        .with_policy(SaveOnlyTheseNames::new(["square"])),
        rematerialize(move |(x, w): (XlaTracer, XlaTracer)| {
            let v = project(staged_inner.call((x, w))?)?;
            Ok(v.mul(&v)?.into_value())
        })
        .with_policy(SaveOnlyTheseNames::new(["square"])),
        array(&[2, 2], &[1.0, -1.0, 2.0, 1.0])?,
    )
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use ryft_core::ResidualSource;

    use super::*;

    #[test]
    fn test_record_residuals() {
        // Saved values are encoded by count and by element counts and bit widths, sorted by element count and then by
        // bit width, so their order and sources do not matter.
        let mut observations = BTreeMap::new();
        record_residuals(
            &mut observations,
            &[
                SavedResidual::new(ArrayType::new_static(DataType::F32, [2, 3]), ResidualSource::Input { index: 0 }),
                SavedResidual::new(
                    ArrayType::new_static(DataType::BF16, [2]),
                    ResidualSource::Operation { name: "dot" },
                ),
                SavedResidual::new(ArrayType::scalar(DataType::F64), ResidualSource::Tag { key: "square".to_owned() }),
                SavedResidual::new(ArrayType::new_static(DataType::F32, [2]), ResidualSource::Constant),
            ],
        )
        .unwrap();
        assert_eq!(
            observations,
            BTreeMap::from([
                ("residual_count", vec![vec![4.0]]),
                ("residual_element_counts", vec![vec![1.0, 2.0, 2.0, 6.0]]),
                ("residual_bit_widths", vec![vec![64.0, 16.0, 32.0, 32.0]]),
            ]),
        );
    }

    #[test]
    fn test_emit_dots_saveable() {
        // `y = x · w = [[6, -2], [0, 2]]` and `b = [208, 8]`, so the output is `b² = [43264, 64]`. Saving dot products
        // saves `y` and `b` together with the inputs.
        let observation = emit_dots_saveable().unwrap();
        let stablehlo = observation.stablehlo.unwrap();
        assert_eq!(observation.case_id, "rematerialization_dots_saveable");
        assert_eq!(
            observation.observations,
            BTreeMap::from([
                ("output", vec![vec![43264.0, 64.0]]),
                ("jvp", vec![vec![134784.0, 192.0]]),
                ("x_gradient", vec![vec![39936.0, 89856.0, -39936.0, 192.0, 0.0, -192.0]]),
                ("w_gradient", vec![vec![44928.0, 4992.0, 89856.0, 9792.0, -44928.0, -5376.0]]),
                ("residual_count", vec![vec![4.0]]),
                ("residual_element_counts", vec![vec![2.0, 4.0, 6.0, 6.0]]),
                ("residual_bit_widths", vec![vec![32.0, 32.0, 32.0, 32.0]]),
            ]),
        );
        assert!(stablehlo.contains("stablehlo.optimization_barrier"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.dot_general"), "{stablehlo}");
    }
}
