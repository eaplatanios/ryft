//! Memory-efficient rescalings that store second moment statistics per axis rather than per element, which are the
//! cores of Adafactor and SM3.
//!
//! [`ScaleByFactoredRms`] factors the second moments of large matrices into row and column statistics, and
//! [`ScaleBySm3`] keeps one accumulator per index of every axis and bounds the second moment of every element by the
//! smallest accumulator of its indices.

use std::marker::PhantomData;

use ryft_core::{
    Add, ComparisonDirection, Concatenate, DataType, Div, Mul, Parameter, Parameterized, ReductionKind, Sub,
};

use crate::macros::parameterized_state;
use crate::{
    Optimizer, OptimizerError, OptimizerValue, UpdateArguments, convert_to, filled_statistics, increment_count,
    initial_count, statistic_data_type, update_moment, validate_decay_rate, validate_non_negative, validate_structure,
    zero_count_like,
};

// TODO(eaplatanios): Review this module.

parameterized_state! {
    /// State of [`ScaleByFactoredRms`], which holds the number of updates applied so far and, with the structure `S` of
    /// the parameters, the row and column statistics of factored leaves and the full statistics of the other leaves.
    /// The statistics that a leaf does not use are vectors of one zero, as in Optax.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct ScaleByFactoredRmsState<P, S> with family ScaleByFactoredRmsStateFamily {
        /// Rank-0 `i32` array that holds the number of updates applied so far.
        count: P,

        /// Moving averages of the squared gradients averaged over the largest axis of factored leaves.
        row_second_moments: S,

        /// Moving averages of the squared gradients averaged over the second largest axis of factored leaves.
        column_second_moments: S,

        /// Moving averages of the squared gradients of leaves that are not factored.
        second_moments: S,
    }
}

impl<P: Parameter, S> ScaleByFactoredRmsState<P, S> {
    /// Returns the rank-0 `i32` array that holds the number of updates applied so far.
    #[inline]
    pub fn count(&self) -> &P {
        &self.count
    }

    /// Returns the moving averages of the squared gradients averaged over the largest axis of factored leaves.
    #[inline]
    pub fn row_second_moments(&self) -> &S {
        &self.row_second_moments
    }

    /// Returns the moving averages of the squared gradients averaged over the second largest axis of factored leaves.
    #[inline]
    pub fn column_second_moments(&self) -> &S {
        &self.column_second_moments
    }

    /// Returns the moving averages of the squared gradients of leaves that are not factored.
    #[inline]
    pub fn second_moments(&self) -> &S {
        &self.second_moments
    }
}

/// Transformation that divides the gradients by the root of factored estimates of their second moments, which is the
/// rescaling of Adafactor of [Shazeer and Stern (2018)](https://arxiv.org/abs/1804.04235) and the analogue of
/// [`optax.scale_by_factored_rms`](
/// https://optax.readthedocs.io/en/latest/api/transformations.html#optax.scale_by_factored_rms).
/// At the step count `t` (i.e., the number of updates before the current one) with the step offset `o`, the decay is
/// `β = 1 - (t - o + 1)^(-d)` with the decay rate `d`. A leaf of rank two or more whose second largest extent is at
/// least the minimum extent to factor is factored over its largest axis `a₀` and its second largest axis `a₁`. For its
/// gradient `g`, the row and column statistics are updated as `r = β · r + (1 - β) · mean(g² + ε, a₀)` and
/// `c = β · c + (1 - β) · mean(g² + ε, a₁)`, and the update is `g · (r / mean(r))^(-1/2) · c^(-1/2)` with `r` and `c`
/// broadcast back along the axes that they were averaged over. Every other leaf keeps full statistics
/// `v = β · v + (1 - β) · (g² + ε)` and is updated as `g · v^(-1/2)`. The defaults are factoring enabled, `d = 0.8`,
/// `o = 0`, a minimum extent to factor of `128`, and `ε = 1e-30`. The custom decay rate functions of Optax are not
/// supported.
#[derive(Clone, Debug, PartialEq)]
pub struct ScaleByFactoredRms {
    /// Whether large leaves are factored.
    factored: bool,

    /// Decay rate `d` of the decay schedule.
    decay_rate: f64,

    /// Step offset `o` of the decay schedule.
    step_offset: usize,

    /// Smallest second largest extent of a leaf that is factored.
    minimum_dimension_size_to_factor: usize,

    /// Term `ε` that is added to the squared gradients.
    epsilon: f64,
}

impl ScaleByFactoredRms {
    /// Creates a new [`ScaleByFactoredRms`] with the default hyperparameters.
    #[inline]
    pub fn new() -> Self {
        Self { factored: true, decay_rate: 0.8, step_offset: 0, minimum_dimension_size_to_factor: 128, epsilon: 1e-30 }
    }

    /// Returns this transformation with factoring enabled when `factored` is `true`.
    #[inline]
    pub fn with_factored(mut self, factored: bool) -> Self {
        self.factored = factored;
        self
    }

    /// Returns this transformation with `decay_rate` as the decay rate of its decay schedule.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `decay_rate` is negative or not finite.
    pub fn with_decay_rate(mut self, decay_rate: f64) -> Result<Self, OptimizerError> {
        validate_non_negative("decay_rate", decay_rate)?;
        self.decay_rate = decay_rate;
        Ok(self)
    }

    /// Returns this transformation with `step_offset` as the step offset of its decay schedule.
    #[inline]
    pub fn with_step_offset(mut self, step_offset: usize) -> Self {
        self.step_offset = step_offset;
        self
    }

    /// Returns this transformation with `minimum_dimension_size_to_factor` as the smallest second largest extent of a
    /// leaf that it factors.
    #[inline]
    pub fn with_minimum_dimension_size_to_factor(mut self, minimum_dimension_size_to_factor: usize) -> Self {
        self.minimum_dimension_size_to_factor = minimum_dimension_size_to_factor;
        self
    }

    /// Returns this transformation with `epsilon` as the term that is added to the squared gradients.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `epsilon` is negative or not finite.
    pub fn with_epsilon(mut self, epsilon: f64) -> Result<Self, OptimizerError> {
        validate_non_negative("epsilon", epsilon)?;
        self.epsilon = epsilon;
        Ok(self)
    }

    /// Returns whether this transformation factors large leaves.
    #[inline]
    pub fn factored(&self) -> bool {
        self.factored
    }

    /// Returns the decay rate `d` of the decay schedule of this transformation.
    #[inline]
    pub fn decay_rate(&self) -> f64 {
        self.decay_rate
    }

    /// Returns the step offset `o` of the decay schedule of this transformation.
    #[inline]
    pub fn step_offset(&self) -> usize {
        self.step_offset
    }

    /// Returns the smallest second largest extent of a leaf that this transformation factors.
    #[inline]
    pub fn minimum_dimension_size_to_factor(&self) -> usize {
        self.minimum_dimension_size_to_factor
    }

    /// Returns the term `ε` that this transformation adds to the squared gradients.
    #[inline]
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// Returns the axes `(a₁, a₀)` over which `value` is factored (i.e., its second largest and its largest axis), or
    /// [`None`] if it is not factored, which is `_factored_dims` of Optax with a stable sort of the extents.
    fn factored_axes<P: OptimizerValue>(&self, value: &P) -> Result<Option<(usize, usize)>, OptimizerError> {
        let dimensions = static_dimensions("ScaleByFactoredRms", value)?;
        if !self.factored || dimensions.len() < 2 {
            return Ok(None);
        }
        let mut sorted_axes = (0..dimensions.len()).collect::<Vec<_>>();
        sorted_axes.sort_by_key(|&axis| dimensions[axis]);
        let (second_largest, largest) = (sorted_axes[sorted_axes.len() - 2], sorted_axes[sorted_axes.len() - 1]);
        if dimensions[second_largest] < self.minimum_dimension_size_to_factor {
            return Ok(None);
        }
        Ok(Some((second_largest, largest)))
    }
}

impl Default for ScaleByFactoredRms {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: OptimizerValue> Optimizer<P> for ScaleByFactoredRms {
    type State<S: Parameterized<P>> = ScaleByFactoredRmsState<P, S>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        let mut rows = Vec::new();
        let mut columns = Vec::new();
        let mut full = Vec::new();
        for parameter in parameters.parameters() {
            let statistic_type = statistic_data_type(parameter.r#type().data_type());
            let parameter = convert_to(parameter, statistic_type)?;
            let unused = convert_to(&zero_count_like(&parameter)?, statistic_type)?.reshape([1])?;
            match self.factored_axes(&parameter)? {
                Some((second_largest, largest)) => {
                    rows.push(parameter.reduce(&[largest], ReductionKind::Sum)?.fill_like(0f64)?);
                    columns.push(parameter.reduce(&[second_largest], ReductionKind::Sum)?.fill_like(0f64)?);
                    full.push(unused);
                }
                None => {
                    rows.push(unused.clone());
                    columns.push(unused);
                    full.push(parameter.fill_like(0f64)?);
                }
            }
        }
        let structure = parameters.parameter_structure();
        Ok(ScaleByFactoredRmsState {
            count: initial_count("ScaleByFactoredRms", parameters)?,
            row_second_moments: S::from_parameters(structure.clone(), rows)?,
            column_second_moments: S::from_parameters(structure.clone(), columns)?,
            second_moments: S::from_parameters(structure, full)?,
        })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        // This follows the operation order of `optax.scale_by_factored_rms` with its default decay rate function, which
        // evaluates the decay at the step count before the update.
        let _ = arguments;
        let paths = gradients.parameter_paths().collect::<Vec<_>>();
        validate_structure("parameters", &paths, parameters)?;
        validate_structure("row_second_moments", &paths, &state.row_second_moments)?;
        validate_structure("column_second_moments", &paths, &state.column_second_moments)?;
        validate_structure("second_moments", &paths, &state.second_moments)?;
        let step = convert_to(&state.count, DataType::F32)?;
        let step = Add::add(&Sub::sub(&step, &step.fill_like(self.step_offset as f64)?)?, &step.fill_like(1f64)?)?;
        let decay = Sub::sub(&step.fill_like(1f64)?, &step.pow(&step.fill_like(-self.decay_rate)?)?)?;
        let structure = gradients.parameter_structure();
        let mut updates = Vec::with_capacity(paths.len());
        let mut rows = Vec::with_capacity(paths.len());
        let mut columns = Vec::with_capacity(paths.len());
        let mut full = Vec::with_capacity(paths.len());
        let statistics = state
            .row_second_moments
            .into_parameters()
            .zip(state.column_second_moments.into_parameters())
            .zip(state.second_moments.into_parameters());
        for (gradient, ((row, column), second_moment)) in gradients.into_parameters().zip(statistics) {
            let data_type = gradient.r#type().data_type();
            let statistic_type = statistic_data_type(data_type);
            let row_type = row.r#type().data_type();
            let column_type = column.r#type().data_type();
            let second_moment_type = second_moment.r#type().data_type();
            let gradient = convert_to(&gradient, statistic_type)?;
            let decay = convert_to(&decay, statistic_type)?;
            let remaining = Sub::sub(&decay.fill_like(1f64)?, &decay)?;
            let squared = Add::add(&Mul::mul(&gradient, &gradient)?, &gradient.fill_like(self.epsilon)?)?;
            let negative_half = gradient.fill_like(-0.5f64)?;
            match self.factored_axes(&gradient)? {
                Some((second_largest, largest)) => {
                    let rank = gradient.r#type().rank();
                    let row = Add::add(
                        &Mul::mul(&decay, &convert_to(&row, statistic_type)?)?,
                        &Mul::mul(&remaining, &squared.reduce(&[largest], ReductionKind::Mean)?)?,
                    )?;
                    let column = Add::add(
                        &Mul::mul(&decay, &convert_to(&column, statistic_type)?)?,
                        &Mul::mul(&remaining, &squared.reduce(&[second_largest], ReductionKind::Mean)?)?,
                    )?;
                    let reduced_axis = if second_largest > largest { second_largest - 1 } else { second_largest };
                    let row_axes = (0..rank - 1).filter(|&axis| axis != reduced_axis).collect::<Vec<_>>();
                    let row_mean = row.reduce(&[reduced_axis], ReductionKind::Mean)?.broadcast_like(&row, &row_axes)?;
                    let row_factor = Div::div(&row, &row_mean)?.pow(&row.fill_like(-0.5f64)?)?;
                    let column_factor = column.pow(&column.fill_like(-0.5f64)?)?;
                    let without = |removed: usize| (0..rank).filter(|&axis| axis != removed).collect::<Vec<_>>();
                    let update = Mul::mul(
                        &Mul::mul(&gradient, &row_factor.broadcast_like(&gradient, &without(largest))?)?,
                        &column_factor.broadcast_like(&gradient, &without(second_largest))?,
                    )?;
                    updates.push(convert_to(&update, data_type)?);
                    rows.push(convert_to(&row, row_type)?);
                    columns.push(convert_to(&column, column_type)?);
                    full.push(convert_to(&second_moment, second_moment_type)?.fill_like(0f64)?);
                }
                None => {
                    let second_moment = Add::add(
                        &Mul::mul(&decay, &convert_to(&second_moment, statistic_type)?)?,
                        &Mul::mul(&remaining, &squared)?,
                    )?;
                    let update = Mul::mul(&gradient, &second_moment.pow(&negative_half)?)?;
                    updates.push(convert_to(&update, data_type)?);
                    rows.push(convert_to(&row, row_type)?.fill_like(0f64)?);
                    columns.push(convert_to(&column, column_type)?.fill_like(0f64)?);
                    full.push(convert_to(&second_moment, second_moment_type)?);
                }
            }
        }
        let state = ScaleByFactoredRmsState {
            count: increment_count(&state.count)?,
            row_second_moments: S::from_parameters(structure.clone(), rows)?,
            column_second_moments: S::from_parameters(structure.clone(), columns)?,
            second_moments: S::from_parameters(structure.clone(), full)?,
        };
        Ok((S::from_parameters(structure, updates)?, state))
    }
}

parameterized_state! {
    /// State of [`ScaleBySm3`], which holds, with the structure `S` of the parameters, the accumulators of every leaf
    /// (concatenated over the axes of the leaf into one vector whose length is the sum of the extents of the leaf) and
    /// the momenta of the updates.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct ScaleBySm3State<P, S> with family ScaleBySm3StateFamily and marker leaf {
        /// Accumulators of every axis of every leaf, concatenated into one vector per leaf.
        accumulators: S,

        /// Momenta of the updates, with the structure of the parameters.
        momenta: S,
    }
}

impl<P: Parameter, S> ScaleBySm3State<P, S> {
    /// Returns the accumulators of every axis of every leaf, concatenated into one vector per leaf.
    #[inline]
    pub fn accumulators(&self) -> &S {
        &self.accumulators
    }

    /// Returns the momenta of the updates, with the structure of the parameters.
    #[inline]
    pub fn momenta(&self) -> &S {
        &self.momenta
    }
}

/// Transformation that keeps one second moment accumulator per index of every axis, which is the rescaling of SM3 of
/// [Anil et al. (2019)](https://arxiv.org/abs/1901.11150) and the analogue of
/// [`optax.scale_by_sm3`](https://optax.readthedocs.io/en/latest/api/transformations.html#optax.scale_by_sm3). For a
/// gradient `g` of a leaf of rank at least one with the accumulators `aᵢ` of its axes `i`, it computes the estimate
/// `v = c · g² + β₂ · minᵢ aᵢ` (where every `aᵢ` is broadcast along the other axes and `c = 1 - β₂`, or `c = 1` when
/// `β₂ = 1`), the update `u = g / √(v + ε)` where `v > 0` and `0` elsewhere, and the momentum
/// `m = (1 - β₁) · u + β₁ · m`, which it returns. The new accumulator of every axis is the maximum of `v` over the
/// other axes (or `v` itself for vectors). Optax stores the accumulators of a leaf as a list of vectors, while this
/// transformation stores them concatenated into one vector. Rank-0 leaves are rejected, as Optax rejects them. The
/// defaults are `β₁ = 0.9`, `β₂ = 1`, and `ε = 1e-8`.
#[derive(Clone, Debug, PartialEq)]
pub struct ScaleBySm3 {
    /// Decay rate `β₁` of the momenta.
    beta1: f64,

    /// Decay rate `β₂` of the accumulators.
    beta2: f64,

    /// Term `ε` that is added to the second moment estimates inside the square root.
    epsilon: f64,
}

impl ScaleBySm3 {
    /// Creates a new [`ScaleBySm3`] with the default hyperparameters.
    #[inline]
    pub fn new() -> Self {
        Self { beta1: 0.9, beta2: 1.0, epsilon: 1e-8 }
    }

    /// Returns this transformation with `beta1` as the decay rate of its momenta.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `beta1` is not in `[0, 1)`.
    pub fn with_beta1(mut self, beta1: f64) -> Result<Self, OptimizerError> {
        validate_decay_rate("beta1", beta1)?;
        self.beta1 = beta1;
        Ok(self)
    }

    /// Returns this transformation with `beta2` as the decay rate of its accumulators.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `beta2` is not in `[0, 1]`.
    pub fn with_beta2(mut self, beta2: f64) -> Result<Self, OptimizerError> {
        if !(0.0..=1.0).contains(&beta2) {
            return Err(OptimizerError::InvalidConfiguration {
                message: format!("`beta2` must be in `[0, 1]` but got `{beta2}`"),
            });
        }
        self.beta2 = beta2;
        Ok(self)
    }

    /// Returns this transformation with `epsilon` as the term that is added to the second moment estimates inside the
    /// square root.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `epsilon` is negative or not finite.
    pub fn with_epsilon(mut self, epsilon: f64) -> Result<Self, OptimizerError> {
        validate_non_negative("epsilon", epsilon)?;
        self.epsilon = epsilon;
        Ok(self)
    }

    /// Returns the decay rate `β₁` of the momenta of this transformation.
    #[inline]
    pub fn beta1(&self) -> f64 {
        self.beta1
    }

    /// Returns the decay rate `β₂` of the accumulators of this transformation.
    #[inline]
    pub fn beta2(&self) -> f64 {
        self.beta2
    }

    /// Returns the term `ε` that this transformation adds to the second moment estimates inside the square root.
    #[inline]
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }
}

impl Default for ScaleBySm3 {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: OptimizerValue> Optimizer<P> for ScaleBySm3 {
    type State<S: Parameterized<P>> = ScaleBySm3State<P, S>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        let accumulators = parameters
            .parameters()
            .map(|parameter| {
                let dimensions = static_dimensions("ScaleBySm3", parameter)?;
                if dimensions.is_empty() {
                    return Err(OptimizerError::StructureMismatch {
                        message: "`ScaleBySm3` does not support rank-0 leaves".to_string(),
                    });
                }
                // The zero accumulator of every axis is the sum over the other axes filled with zeros.
                let parameter = convert_to(parameter, statistic_data_type(parameter.r#type().data_type()))?;
                let rank = dimensions.len();
                let zeros = (0..rank)
                    .map(|axis| {
                        let other_axes = (0..rank).filter(|&other| other != axis).collect::<Vec<_>>();
                        Ok(parameter.reduce(&other_axes, ReductionKind::Sum)?.fill_like(0f64)?)
                    })
                    .collect::<Result<Vec<_>, OptimizerError>>()?;
                Ok(Concatenate::concatenate(&zeros, 0)?)
            })
            .collect::<Result<Vec<_>, OptimizerError>>()?;
        Ok(ScaleBySm3State {
            accumulators: S::from_parameters(parameters.parameter_structure(), accumulators)?,
            momenta: filled_statistics(parameters, 0.0, None)?,
            leaf: PhantomData,
        })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        // This follows the operation order of `optax.scale_by_sm3`.
        let _ = arguments;
        let paths = gradients.parameter_paths().collect::<Vec<_>>();
        validate_structure("parameters", &paths, parameters)?;
        validate_structure("accumulators", &paths, &state.accumulators)?;
        validate_structure("momenta", &paths, &state.momenta)?;
        let gradient_coefficient = if self.beta2 == 1.0 { 1.0 } else { 1.0 - self.beta2 };
        let structure = gradients.parameter_structure();
        let mut updates = Vec::with_capacity(paths.len());
        let mut accumulators = Vec::with_capacity(paths.len());
        let mut momenta = Vec::with_capacity(paths.len());
        let statistics = state.accumulators.into_parameters().zip(state.momenta.into_parameters());
        for (gradient, (accumulator, momentum)) in gradients.into_parameters().zip(statistics) {
            let data_type = gradient.r#type().data_type();
            let statistic_type = statistic_data_type(data_type);
            let accumulator_type = accumulator.r#type().data_type();
            let momentum_type = momentum.r#type().data_type();
            let gradient = convert_to(&gradient, statistic_type)?;
            let accumulator = convert_to(&accumulator, statistic_type)?;
            let dimensions = static_dimensions("ScaleBySm3", &gradient)?;
            let rank = dimensions.len();
            let mut offset = 0;
            let mut minimum: Option<P> = None;
            for (axis, &dimension) in dimensions.iter().enumerate() {
                let axis_accumulator =
                    accumulator.slice(&[offset], &[offset + dimension], &[1])?.broadcast_like(&gradient, &[axis])?;
                offset += dimension;
                minimum = Some(match minimum {
                    None => axis_accumulator,
                    Some(minimum) => minimum.min(&axis_accumulator)?,
                });
            }
            let minimum = minimum.ok_or_else(|| OptimizerError::StructureMismatch {
                message: "`ScaleBySm3` does not support rank-0 leaves".to_string(),
            })?;
            let estimate = Add::add(
                &Mul::mul(&gradient.fill_like(gradient_coefficient)?, &Mul::mul(&gradient, &gradient)?)?,
                &Mul::mul(&gradient.fill_like(self.beta2)?, &minimum)?,
            )?;
            let zero = gradient.fill_like(0f64)?;
            let inverse = estimate
                .compare(&zero, ComparisonDirection::GreaterThan)?
                .select(&Add::add(&estimate, &gradient.fill_like(self.epsilon)?)?.rsqrt()?, &zero)?;
            let update = Mul::mul(&gradient, &inverse)?;
            let momentum = update_moment(&update, &convert_to(&momentum, statistic_type)?, self.beta1)?;
            let accumulator = if rank < 2 {
                estimate
            } else {
                let maxima = (0..rank)
                    .map(|axis| {
                        let other_axes = (0..rank).filter(|&other| other != axis).collect::<Vec<_>>();
                        Ok(estimate.reduce(&other_axes, ReductionKind::Max)?)
                    })
                    .collect::<Result<Vec<_>, OptimizerError>>()?;
                Concatenate::concatenate(&maxima, 0)?
            };
            updates.push(convert_to(&momentum, data_type)?);
            accumulators.push(convert_to(&accumulator, accumulator_type)?);
            momenta.push(convert_to(&momentum, momentum_type)?);
        }
        let state = ScaleBySm3State {
            accumulators: S::from_parameters(structure.clone(), accumulators)?,
            momenta: S::from_parameters(structure.clone(), momenta)?,
            leaf: PhantomData,
        };
        Ok((S::from_parameters(structure, updates)?, state))
    }
}

/// Returns the static extents of `value`, or an error naming `optimizer` when its shape is dynamic, which the
/// transformations of this module need to choose their per-axis statistics.
fn static_dimensions<P: OptimizerValue>(optimizer: &str, value: &P) -> Result<Vec<usize>, OptimizerError> {
    value.r#type().static_shape().map(|shape| shape.dimensions().to_vec()).ok_or_else(|| {
        OptimizerError::StructureMismatch { message: format!("`{optimizer}` requires leaves with static shapes") }
    })
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use ryft_core::Array;

    use crate::tests::{assert_parity, parity_updates, state_paths, vector_updates};

    use super::*;

    #[test]
    fn test_scale_by_factored_rms_new() {
        let factored = ScaleByFactoredRms::new();
        assert!(factored.factored());
        assert_eq!(
            (
                factored.decay_rate(),
                factored.step_offset(),
                factored.minimum_dimension_size_to_factor(),
                factored.epsilon()
            ),
            (0.8, 0, 128, 1e-30),
        );
        assert_eq!(ScaleByFactoredRms::default(), factored);
        let factored = factored
            .with_factored(false)
            .with_decay_rate(0.5)
            .unwrap()
            .with_step_offset(2)
            .with_minimum_dimension_size_to_factor(4)
            .with_epsilon(0.0)
            .unwrap();
        assert!(!factored.factored());
        assert_eq!(
            (
                factored.decay_rate(),
                factored.step_offset(),
                factored.minimum_dimension_size_to_factor(),
                factored.epsilon()
            ),
            (0.5, 2, 4, 0.0),
        );
        assert!(matches!(
            ScaleByFactoredRms::new().with_decay_rate(-0.5),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`decay_rate` must be finite and non-negative but got `-0.5`",
        ));
    }

    #[test]
    fn test_scale_by_factored_rms_initialize() {
        assert_eq!(
            state_paths(&ScaleByFactoredRms::new()),
            vec![
                "$.count",
                "$.row_second_moments[\"bias\"]",
                "$.row_second_moments[\"weights\"]",
                "$.column_second_moments[\"bias\"]",
                "$.column_second_moments[\"weights\"]",
                "$.second_moments[\"bias\"]",
                "$.second_moments[\"weights\"]",
            ],
        );

        // A `2 × 3` matrix is factored over its rows and columns once the minimum extent to factor is `2`.
        let parameters = vec![Array::matrix(2, 3, vec![1.0f32; 6]).unwrap()];
        let state = ScaleByFactoredRms::new().with_minimum_dimension_size_to_factor(2).initialize(&parameters).unwrap();
        assert_eq!(state.row_second_moments(), &vec![Array::vector(vec![0.0f32, 0.0]).unwrap()]);
        assert_eq!(state.column_second_moments(), &vec![Array::vector(vec![0.0f32, 0.0, 0.0]).unwrap()]);
        assert_eq!(state.second_moments(), &vec![Array::vector(vec![0.0f32]).unwrap()]);
    }

    #[test]
    fn test_scale_by_factored_rms_update() {
        // A vector is not factored. The decay at the first step is `1 - 1^(-0.8) = 0`, so `v = 2² = 4` and the update
        // is `2 · 4^(-1/2)`. The decay at the second step is `β = 1 - 2^(-0.8)`, so `v = β · 4 + (1 - β) · 1`.
        let factored = ScaleByFactoredRms::new();
        let decay = 1.0f32 - 2.0f32.powf(-0.8);
        let second_moment = decay * 4.0 + (1.0 - decay) * 1.0;
        assert_eq!(
            vector_updates(&factored, &[0.0], &[&[2.0], &[1.0]]),
            vec![vec![2.0 * 4.0f32.powf(-0.5)], vec![1.0 * second_moment.powf(-0.5)],],
        );

        // The `2 × 3` matrix `[[1, 1, 1], [2, 2, 2]]` has the row statistics `[1, 4]` (whose mean is `2.5`) and the
        // column statistics `[2.5, 2.5, 2.5]` at the first step.
        let factored = factored.with_minimum_dimension_size_to_factor(2);
        let gradients = vec![Array::matrix(2, 3, vec![1.0f32, 1.0, 1.0, 2.0, 2.0, 2.0]).unwrap()];
        let state = factored.initialize(&gradients).unwrap();
        let (updates, state) = factored.update(gradients.clone(), state, &gradients, &UpdateArguments::new()).unwrap();
        let first = 1.0 * (1.0f32 / 2.5).powf(-0.5) * 2.5f32.powf(-0.5);
        let second = 2.0 * (4.0f32 / 2.5).powf(-0.5) * 2.5f32.powf(-0.5);
        assert_eq!(updates, vec![Array::matrix(2, 3, vec![first, first, first, second, second, second]).unwrap()]);
        assert_eq!(state.row_second_moments(), &vec![Array::vector(vec![1.0f32, 4.0]).unwrap()]);
        assert_eq!(state.count(), &Array::scalar(1i32).unwrap());
    }

    #[test]
    fn test_scale_by_sm3_new() {
        let sm3 = ScaleBySm3::new();
        assert_eq!((sm3.beta1(), sm3.beta2(), sm3.epsilon()), (0.9, 1.0, 1e-8));
        assert_eq!(ScaleBySm3::default(), sm3);
        let sm3 = sm3.with_beta1(0.5).unwrap().with_beta2(0.5).unwrap().with_epsilon(0.0).unwrap();
        assert_eq!((sm3.beta1(), sm3.beta2(), sm3.epsilon()), (0.5, 0.5, 0.0));
        assert!(matches!(
            ScaleBySm3::new().with_beta2(1.5),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`beta2` must be in `[0, 1]` but got `1.5`",
        ));
    }

    #[test]
    fn test_scale_by_sm3_initialize() {
        let parameters = vec![Array::matrix(2, 3, vec![1.0f32; 6]).unwrap()];
        let state = ScaleBySm3::new().initialize(&parameters).unwrap();
        assert_eq!(state.accumulators(), &vec![Array::vector(vec![0.0f32; 5]).unwrap()]);
        assert_eq!(
            state.named_parameters().map(|(path, _)| path.to_string()).collect::<Vec<_>>(),
            vec!["$.accumulators[0]", "$.momenta[0]"],
        );
        assert!(matches!(
            Optimizer::<Array>::initialize(&ScaleBySm3::new(), &vec![Array::scalar(1.0f32).unwrap()]),
            Err(OptimizerError::StructureMismatch { message })
                if message == "`ScaleBySm3` does not support rank-0 leaves",
        ));
    }

    #[test]
    fn test_scale_by_sm3_update() {
        // With `β₁ = 0.5`, `β₂ = 1`, and `ε = 0`, the gradient `[2, 0]` of a vector gives `v = [4, 0]` and
        // `u = [1, 0]`, so the momentum is `[0.5, 0]`. The gradient `[1, 3]` gives `v = [5, 9]` and
        // `u = [1 / √5, 3 / √9]`.
        let sm3 = ScaleBySm3::new().with_beta1(0.5).unwrap().with_epsilon(0.0).unwrap();
        let second = [5.0f32.sqrt().recip() * 1.0, 9.0f32.sqrt().recip() * 3.0];
        assert_eq!(
            vector_updates(&sm3, &[0.0, 0.0], &[&[2.0, 0.0], &[1.0, 3.0]]),
            vec![vec![0.5, 0.0], vec![0.5 * second[0] + 0.5 * 0.5, 0.5 * second[1] + 0.5 * 0.0],],
        );

        // For a matrix, the new accumulators are the maxima of `v = g²` over the rows and over the columns.
        let gradients = vec![Array::matrix(2, 2, vec![1.0f32, 2.0, 3.0, 4.0]).unwrap()];
        let state = sm3.initialize(&gradients).unwrap();
        let (_, state) = sm3.update(gradients.clone(), state, &gradients, &UpdateArguments::new()).unwrap();
        assert_eq!(state.accumulators(), &vec![Array::vector(vec![4.0f32, 16.0, 9.0, 16.0]).unwrap()]);
    }

    #[test]
    fn test_scale_by_factored_rms_optax_parity() {
        // Optax 9ecde04b: `optax.scale_by_factored_rms(factored=True, decay_rate=0.8, step_offset=0,
        // min_dim_size_to_factor=2, epsilon=1e-3)`.
        const SCALE_BY_FACTORED_RMS: [[f32; 9]; 4] = [
            [0.95346254, -0.98772955, 0.9944903, 1.6524092, -0.405142, 0.65414625, -0.68216753, 1.1150374, -1.0802103],
            [-0.6111948, 1.0765979, 0.47113204, 0.8615747, 1.1057113, -1.1032714, 0.34659514, -0.5930755, 0.7766938],
            [1.469096, -0.52632004, -0.9603686, -0.67311543, 0.18144874, 1.2907703, 0.9683356, 1.2529438, -0.2475849],
            [0.9821226, 0.83927125, -1.324744, 0.42305085, -1.2132334, 0.17314175, -1.4769696, 0.6275085, 0.8059714],
        ];
        let optimizer = ScaleByFactoredRms::new()
            .with_factored(true)
            .with_decay_rate(0.8)
            .unwrap()
            .with_step_offset(0)
            .with_minimum_dimension_size_to_factor(2)
            .with_epsilon(1e-3)
            .unwrap();
        assert_parity(&parity_updates(&optimizer), &SCALE_BY_FACTORED_RMS);

        // Optax 9ecde04b: `optax.scale_by_factored_rms(factored=True, decay_rate=0.8, step_offset=0,
        // min_dim_size_to_factor=128, epsilon=1e-3)`.
        const SCALE_BY_FACTORED_RMS_UNFACTORED: [[f32; 9]; 4] = [
            [0.95346254, -0.98772955, 0.9944903, 0.99688953, -0.95346254, 0.98772955, -0.9944903, 0.998006, -0.9986141],
            [-0.6111948, 1.0765979, 0.47113204, 0.6591008, 1.257136, -1.2064737, 0.47113204, -0.5538771, 0.73853457],
            [1.469096, -0.52632004, -0.9603686, -0.59478176, 0.26820537, 1.1666507, 1.0888884, 1.057407, -0.27126348],
            [0.9821226, 0.83927125, -1.324744, 0.4649932, -1.2993433, 0.15744208, -1.4058808, 0.6046225, 0.8621257],
        ];
        let optimizer = ScaleByFactoredRms::new()
            .with_factored(true)
            .with_decay_rate(0.8)
            .unwrap()
            .with_step_offset(0)
            .with_minimum_dimension_size_to_factor(128)
            .with_epsilon(1e-3)
            .unwrap();
        assert_parity(&parity_updates(&optimizer), &SCALE_BY_FACTORED_RMS_UNFACTORED);
    }

    #[test]
    fn test_scale_by_sm3_optax_parity() {
        // Optax 9ecde04b: `optax.scale_by_sm3(b1=0.8, b2=0.95, eps=1e-3)`.
        const SCALE_BY_SM3: [[f32; 9]; 4] = [
            [
                0.5163978,
                -0.73029673,
                0.8090398,
                0.84327406,
                -0.5163978,
                0.73029673,
                -0.8090398,
                0.86066294,
                -0.87057155,
            ],
            [
                0.16311821,
                0.059919238,
                0.9104125,
                1.0631336,
                0.11110428,
                -0.036683083,
                -0.43757483,
                0.36056235,
                -0.24671248,
            ],
            [0.900612, -0.20456749, 0.26756197, 0.5676013, 0.17248935, 0.5294011, 0.084394336, 0.82560164, -0.327659],
            [1.1729221, 0.19150463, -0.39214313, 0.6286038, -0.23684613, 0.48699185, -0.5290771, 0.9235357, 0.10345769],
        ];
        let optimizer =
            ScaleBySm3::new().with_beta1(0.8).unwrap().with_beta2(0.95).unwrap().with_epsilon(1e-3).unwrap();
        assert_parity(&parity_updates(&optimizer), &SCALE_BY_SM3);
    }
}
