//! Momentum accumulators, which replace the updates with a running sum or a running average of the updates.
//!
//! [`Trace`] accumulates a decaying sum of the updates, which is the momentum of SGD (optionally with Nesterov
//! momentum), and [`Ema`] accumulates an exponential moving average of the updates, which can be corrected for its
//! initialization bias.

use std::marker::PhantomData;

use ryft_core::{Add, DataType, Mul, Parameter, Parameterized};

use crate::macros::parameterized_state;
use crate::{
    Optimizer, OptimizerError, OptimizerValue, UpdateArguments, bias_corrected, bias_correction, convert_to,
    filled_statistics, increment_count, initial_count, statistic_data_type, update_moment, validate_decay_rate,
    validate_finite, validate_floating_point, validate_structure,
};

// TODO(eaplatanios): Review this module.

parameterized_state! {
    /// State of [`Trace`], which holds the accumulated updates with the structure `S` of the parameters (e.g.,
    /// `$.trace.weights` for the parameter at `$.weights`).
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct TraceState<P, S> with family TraceStateFamily and marker leaf {
        /// Decaying sums of the updates, with the structure of the parameters.
        trace: S,
    }
}

impl<P: Parameter, S> TraceState<P, S> {
    /// Returns the decaying sums of the updates, with the structure of the parameters.
    #[inline]
    pub fn trace(&self) -> &S {
        &self.trace
    }
}

/// Transformation that accumulates a decaying sum of the updates, which is the momentum of SGD and the analogue of
/// [`optax.trace`](https://optax.readthedocs.io/en/latest/api/transformations.html#optax.trace). For an update `g` and
/// the decay `μ`, it updates the trace as `t = g + μ · t` and returns `t`, or `g + μ · t` with Nesterov momentum
/// ([Sutskever et al., 2013](http://proceedings.mlr.press/v28/sutskever13.html)).
#[derive(Clone, Debug, PartialEq)]
pub struct Trace {
    /// Decay `μ` of the trace.
    decay: f64,

    /// Whether the updates use Nesterov momentum.
    nesterov: bool,

    /// Data type in which the trace is stored, or [`None`] for the statistic data type of each leaf.
    accumulator_data_type: Option<DataType>,
}

impl Trace {
    /// Creates a new [`Trace`] with the decay `decay` and without Nesterov momentum.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `decay` is not finite.
    pub fn new(decay: f64) -> Result<Self, OptimizerError> {
        validate_finite("decay", decay)?;
        Ok(Self { decay, nesterov: false, accumulator_data_type: None })
    }

    /// Returns this transformation with Nesterov momentum when `nesterov` is `true`.
    #[inline]
    pub fn with_nesterov(mut self, nesterov: bool) -> Self {
        self.nesterov = nesterov;
        self
    }

    /// Returns this transformation with its trace stored in `data_type` (i.e., the `accumulator_dtype` of Optax). The
    /// trace is still computed in the statistic data type of each leaf and converted to `data_type` when it is stored.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `data_type` is not a floating-point data type.
    pub fn with_accumulator_data_type(mut self, data_type: DataType) -> Result<Self, OptimizerError> {
        validate_floating_point("accumulator_data_type", data_type)?;
        self.accumulator_data_type = Some(data_type);
        Ok(self)
    }

    /// Returns the decay `μ` of the trace of this transformation.
    #[inline]
    pub fn decay(&self) -> f64 {
        self.decay
    }

    /// Returns whether the updates of this transformation use Nesterov momentum.
    #[inline]
    pub fn nesterov(&self) -> bool {
        self.nesterov
    }

    /// Returns the data type in which this transformation stores its trace, or [`None`] if it stores it in the
    /// statistic data type of each leaf.
    #[inline]
    pub fn accumulator_data_type(&self) -> Option<DataType> {
        self.accumulator_data_type
    }
}

impl<P: OptimizerValue> Optimizer<P> for Trace {
    type State<S: Parameterized<P>> = TraceState<P, S>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        Ok(TraceState { trace: filled_statistics(parameters, 0.0, self.accumulator_data_type)?, leaf: PhantomData })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        // This follows the operation order of `optax.trace`.
        let _ = arguments;
        let paths = gradients.parameter_paths().collect::<Vec<_>>();
        validate_structure("parameters", &paths, parameters)?;
        validate_structure("trace", &paths, &state.trace)?;
        let structure = gradients.parameter_structure();
        let mut updates = Vec::with_capacity(paths.len());
        let mut traces = Vec::with_capacity(paths.len());
        for (gradient, trace) in gradients.into_parameters().zip(state.trace.into_parameters()) {
            let data_type = gradient.r#type().data_type();
            let statistic_type = statistic_data_type(data_type);
            let trace_type = trace.r#type().data_type();
            let gradient = convert_to(&gradient, statistic_type)?;
            let decay = gradient.fill_like(self.decay)?;
            let trace = Add::add(&gradient, &Mul::mul(&decay, &convert_to(&trace, statistic_type)?)?)?;
            let update = if self.nesterov { Add::add(&gradient, &Mul::mul(&decay, &trace)?)? } else { trace.clone() };
            updates.push(convert_to(&update, data_type)?);
            traces.push(convert_to(&trace, trace_type)?);
        }
        let state = TraceState { trace: S::from_parameters(structure.clone(), traces)?, leaf: PhantomData };
        Ok((S::from_parameters(structure, updates)?, state))
    }
}

parameterized_state! {
    /// State of [`Ema`], which holds the number of updates applied so far and the exponential moving averages of the
    /// updates, with the structure `S` of the parameters.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct EmaState<P, S> with family EmaStateFamily {
        /// Rank-0 `i32` array that holds the number of updates applied so far.
        count: P,

        /// Exponential moving averages of the updates, with the structure of the parameters.
        ema: S,
    }
}

impl<P: Parameter, S> EmaState<P, S> {
    /// Returns the rank-0 `i32` array that holds the number of updates applied so far.
    #[inline]
    pub fn count(&self) -> &P {
        &self.count
    }

    /// Returns the exponential moving averages of the updates, with the structure of the parameters.
    #[inline]
    pub fn ema(&self) -> &S {
        &self.ema
    }
}

/// Transformation that replaces the updates with their exponential moving average, which is the analogue of
/// [`optax.ema`](https://optax.readthedocs.io/en/latest/api/transformations.html#optax.ema). For an update `g`, the
/// decay `β`, and the step count `t` (i.e., the number of updates including the current one), it updates the average as
/// `m = (1 - β) · g + β · m` and returns `m / (1 - βᵗ)`, or `m` itself when debiasing is disabled.
#[derive(Clone, Debug, PartialEq)]
pub struct Ema {
    /// Decay `β` of the moving average.
    decay: f64,

    /// Whether the moving average is corrected for its initialization bias.
    debias: bool,

    /// Data type in which the moving average is stored, or [`None`] for the statistic data type of each leaf.
    accumulator_data_type: Option<DataType>,
}

impl Ema {
    /// Creates a new [`Ema`] with the decay `decay` that corrects its moving average for its initialization bias.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `decay` is not in `[0, 1)`.
    pub fn new(decay: f64) -> Result<Self, OptimizerError> {
        validate_decay_rate("decay", decay)?;
        Ok(Self { decay, debias: true, accumulator_data_type: None })
    }

    /// Returns this transformation with the bias correction of its moving average enabled when `debias` is `true`.
    #[inline]
    pub fn with_debias(mut self, debias: bool) -> Self {
        self.debias = debias;
        self
    }

    /// Returns this transformation with its moving average stored in `data_type` (refer to
    /// [`Trace::with_accumulator_data_type`]).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `data_type` is not a floating-point data type.
    pub fn with_accumulator_data_type(mut self, data_type: DataType) -> Result<Self, OptimizerError> {
        validate_floating_point("accumulator_data_type", data_type)?;
        self.accumulator_data_type = Some(data_type);
        Ok(self)
    }

    /// Returns the decay `β` of the moving average of this transformation.
    #[inline]
    pub fn decay(&self) -> f64 {
        self.decay
    }

    /// Returns whether this transformation corrects its moving average for its initialization bias.
    #[inline]
    pub fn debias(&self) -> bool {
        self.debias
    }

    /// Returns the data type in which this transformation stores its moving average, or [`None`] if it stores it in
    /// the statistic data type of each leaf.
    #[inline]
    pub fn accumulator_data_type(&self) -> Option<DataType> {
        self.accumulator_data_type
    }
}

impl<P: OptimizerValue> Optimizer<P> for Ema {
    type State<S: Parameterized<P>> = EmaState<P, S>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        Ok(EmaState {
            count: initial_count("Ema", parameters)?,
            ema: filled_statistics(parameters, 0.0, self.accumulator_data_type)?,
        })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        // This follows the operation order of `optax.ema`.
        let _ = arguments;
        let paths = gradients.parameter_paths().collect::<Vec<_>>();
        validate_structure("parameters", &paths, parameters)?;
        validate_structure("ema", &paths, &state.ema)?;
        let count = increment_count(&state.count)?;
        let correction = bias_correction(&convert_to(&count, DataType::F32)?, self.decay)?;
        let structure = gradients.parameter_structure();
        let mut updates = Vec::with_capacity(paths.len());
        let mut averages = Vec::with_capacity(paths.len());
        for (gradient, average) in gradients.into_parameters().zip(state.ema.into_parameters()) {
            let data_type = gradient.r#type().data_type();
            let statistic_type = statistic_data_type(data_type);
            let average_type = average.r#type().data_type();
            let gradient = convert_to(&gradient, statistic_type)?;
            let average = update_moment(&gradient, &convert_to(&average, statistic_type)?, self.decay)?;
            let update = if self.debias { bias_corrected(&average, &correction)? } else { average.clone() };
            updates.push(convert_to(&update, data_type)?);
            averages.push(convert_to(&average, average_type)?);
        }
        let state = EmaState { count, ema: S::from_parameters(structure.clone(), averages)? };
        Ok((S::from_parameters(structure, updates)?, state))
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use ryft_core::{Array, Typed};

    use crate::tests::{assert_parity, parity_updates, state_paths, vector_updates};

    use super::*;

    #[test]
    fn test_trace_new() {
        let trace = Trace::new(0.9).unwrap();
        assert_eq!(trace.decay(), 0.9);
        assert!(!trace.nesterov());
        assert_eq!(trace.accumulator_data_type(), None);
        assert!(matches!(
            Trace::new(f64::NAN),
            Err(OptimizerError::InvalidConfiguration { message }) if message == "`decay` must be finite but got `NaN`",
        ));
    }

    #[test]
    fn test_trace_with_nesterov() {
        assert!(Trace::new(0.9).unwrap().with_nesterov(true).nesterov());
    }

    #[test]
    fn test_trace_with_accumulator_data_type() {
        let trace = Trace::new(0.5).unwrap().with_accumulator_data_type(DataType::BF16).unwrap();
        assert_eq!(trace.accumulator_data_type(), Some(DataType::BF16));
        let parameters = vec![Array::vector(vec![1.0f32]).unwrap()];
        assert_eq!(trace.initialize(&parameters).unwrap().trace()[0].r#type().data_type(), DataType::BF16);
        assert!(matches!(
            Trace::new(0.5).unwrap().with_accumulator_data_type(DataType::Boolean),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`accumulator_data_type` must be a floating-point data type but got `bool`",
        ));
    }

    #[test]
    fn test_trace_initialize() {
        assert_eq!(state_paths(&Trace::new(0.9).unwrap()), vec!["$.trace[\"bias\"]", "$.trace[\"weights\"]"]);
    }

    #[test]
    fn test_trace_update() {
        // With `μ = 0.5`, the gradients `2` and `1` give the traces `2` and `1 + 0.5 · 2 = 2`, and with Nesterov
        // momentum the updates are `2 + 0.5 · 2 = 3` and `1 + 0.5 · 2 = 2`.
        let trace = Trace::new(0.5).unwrap();
        assert_eq!(vector_updates(&trace, &[0.0], &[&[2.0], &[1.0]]), vec![vec![2.0], vec![2.0]]);
        assert_eq!(vector_updates(&trace.with_nesterov(true), &[0.0], &[&[2.0], &[1.0]]), vec![vec![3.0], vec![2.0]]);
    }

    #[test]
    fn test_ema_new() {
        let ema = Ema::new(0.9).unwrap();
        assert_eq!(ema.decay(), 0.9);
        assert!(ema.debias());
        assert_eq!(ema.accumulator_data_type(), None);
        assert!(matches!(
            Ema::new(1.0),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`decay` must be in `[0, 1)` but got `1`",
        ));
    }

    #[test]
    fn test_ema_with_debias() {
        assert!(!Ema::new(0.9).unwrap().with_debias(false).debias());
    }

    #[test]
    fn test_ema_with_accumulator_data_type() {
        let ema = Ema::new(0.9).unwrap().with_accumulator_data_type(DataType::F64).unwrap();
        assert_eq!(ema.accumulator_data_type(), Some(DataType::F64));
    }

    #[test]
    fn test_ema_initialize() {
        assert_eq!(state_paths(&Ema::new(0.9).unwrap()), vec!["$.count", "$.ema[\"bias\"]", "$.ema[\"weights\"]"],);
    }

    #[test]
    fn test_ema_update() {
        // With `β = 0.5`, the gradients `2` and `1` give the averages `1` and `0.5 + 0.5 = 1`, which are debiased to
        // `1 / 0.5 = 2` and `1 / 0.75`.
        let ema = Ema::new(0.5).unwrap();
        assert_eq!(vector_updates(&ema, &[0.0], &[&[2.0], &[1.0]]), vec![vec![2.0], vec![1.0f32 / 0.75]]);
        assert_eq!(vector_updates(&ema.with_debias(false), &[0.0], &[&[2.0], &[1.0]]), vec![vec![1.0], vec![1.0]]);
    }

    #[test]
    fn test_trace_optax_parity() {
        // Optax 9ecde04b: `optax.trace(decay=0.8)`.
        const TRACE: [[f32; 9]; 4] = [
            [0.1, -0.2, 0.3, 0.4, -0.1, 0.2, -0.3, 0.5, -0.6],
            [0.030000005, 0.08999999, 0.34, 0.52000004, 0.22, -0.24, -0.14000002, 0.2, -0.13000003],
            [0.324, -0.028000012, 0.07200001, 0.26600003, 0.226, 0.258, 0.13799998, 0.56, -0.20400003],
            [0.45920002, 0.1276, -0.29239997, 0.31280002, -0.119200006, 0.2564, -0.3396, 0.648, 0.13679999],
        ];
        let optimizer = Trace::new(0.8).unwrap();
        assert_parity(&parity_updates(&optimizer), &TRACE);

        // Optax 9ecde04b: `optax.trace(decay=0.8, nesterov=True)`.
        const TRACE_NESTEROV: [[f32; 9]; 4] = [
            [0.18, -0.36, 0.54, 0.72, -0.18, 0.36, -0.54, 0.9, -1.08],
            [-0.025999997, 0.322, 0.372, 0.61600006, 0.476, -0.592, -0.012000009, -0.03999999, 0.24599996],
            [0.55920005, -0.122400016, -0.1424, 0.06280002, 0.2308, 0.65639997, 0.3604, 0.84800005, -0.2632],
            [0.56736004, 0.25208002, -0.58392, 0.35024002, -0.39536002, 0.25512, -0.72168, 0.7184, 0.40944],
        ];
        let optimizer = Trace::new(0.8).unwrap().with_nesterov(true);
        assert_parity(&parity_updates(&optimizer), &TRACE_NESTEROV);
    }

    #[test]
    fn test_ema_optax_parity() {
        // Optax 9ecde04b: `optax.ema(decay=0.8, debias=True)`.
        const EMA: [[f32; 9]; 4] = [
            [
                0.10000002,
                -0.20000003,
                0.30000004,
                0.40000007,
                -0.10000002,
                0.20000003,
                -0.30000004,
                0.50000006,
                -0.6000001,
            ],
            [
                0.01666667,
                0.050000004,
                0.18888892,
                0.28888893,
                0.12222224,
                -0.13333336,
                -0.077777795,
                0.111111134,
                -0.072222255,
            ],
            [
                0.1327869,
                -0.011475414,
                0.029508201,
                0.109016396,
                0.09262296,
                0.10573769,
                0.056557376,
                0.22950822,
                -0.08360658,
            ],
            [
                0.15555558,
                0.043224934,
                -0.09905149,
                0.10596206,
                -0.040379405,
                0.086856365,
                -0.11504065,
                0.21951221,
                0.046341456,
            ],
        ];
        let optimizer = Ema::new(0.8).unwrap().with_debias(true);
        assert_parity(&parity_updates(&optimizer), &EMA);

        // Optax 9ecde04b: `optax.ema(decay=0.8, debias=False)`.
        const EMA_BIASED: [[f32; 9]; 4] = [
            [
                0.020000001,
                -0.040000003,
                0.060000002,
                0.080000006,
                -0.020000001,
                0.040000003,
                -0.060000002,
                0.1,
                -0.120000005,
            ],
            [0.006, 0.018, 0.068, 0.104, 0.044, -0.048000004, -0.028000003, 0.040000003, -0.026000008],
            [0.0648, -0.0056000017, 0.014400002, 0.0532, 0.0452, 0.05159999, 0.027599998, 0.112, -0.04080001],
            [0.09184001, 0.02552, -0.05848, 0.06256, -0.023839999, 0.051279996, -0.06792, 0.1296, 0.027359996],
        ];
        let optimizer = Ema::new(0.8).unwrap().with_debias(false);
        assert_parity(&parity_updates(&optimizer), &EMA_BIASED);
    }
}
