//! Transformations that orthogonalize the momentum of matrix parameters.
//!
//!   - **Muon:** [`ScaleByMuon`] orthogonalizes the momentum of parameter matrices with Newton–Schulz iterations
//!     (refer to [`Muon`](crate::Muon), which applies it to the matrix leaves and AdamW to all other leaves).
//!   - **Dimuon:** [`ScaleByDimuon`] orthogonalizes the row and column conditioned momentum of LoRA factors (refer to
//!     [`Dimuon`](crate::Dimuon), which adds global gradient clipping and a learning rate). It is an unofficial
//!     reconstruction of the optimizer of Thinking Machines, whose formulation is unpublished.
//!
//! [`ScaleByMuon`] and [`ScaleByDimuon`] arrange every leaf as a batch of matrices (i.e., an array of shape
//! `[batch, rows, columns]`) and share the private helpers of this module for the products, transpositions, and
//! Frobenius norms of such batches, and for the polynomial iterations that drive the singular values of the matrices
//! towards one.

use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;

use ryft_core::{
    Add, ComparisonDirection, Concatenate, DataType, Dimension, Div, DotDimensionNumbers, Mul, Parameter,
    ParameterPath, Parameterized, ReductionKind, Shape,
};

use crate::macros::parameterized_state;
use crate::{
    Optimizer, OptimizerError, OptimizerValue, UpdateArguments, all_axes, bias_corrected, bias_correction, convert_to,
    filled_statistics, increment_count, initial_count, statistic_data_type, update_moment, validate_decay_rate,
    validate_finite, validate_floating_point, validate_non_negative, validate_structure, zero_count_like,
};

// TODO(eaplatanios): Review this module.

/// Axes of a parameter that [`ScaleByMuon`] treats as the rows (i.e., the reduction axes, whose product is the fan in)
/// and the columns (i.e., the output axes, whose product is the fan out) of the matrices that it orthogonalizes, which
/// is the analogue of
/// [`optax.contrib.MuonDimensionNumbers`](
/// https://optax.readthedocs.io/en/latest/api/contrib.html#optax.contrib.MuonDimensionNumbers).
/// All other axes are batch axes, along which every matrix is orthogonalized separately. Negative axes count from the
/// last axis. The default is the reduction axis `0` and the output axis `1` of a rank-2 parameter.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MuonDimensionNumbers {
    /// Axes whose extents multiply to the number of rows.
    reduction_axes: Vec<isize>,

    /// Axes whose extents multiply to the number of columns.
    output_axes: Vec<isize>,
}

impl MuonDimensionNumbers {
    /// Creates new [`MuonDimensionNumbers`] with the reduction axes `reduction_axes` and the output axes `output_axes`.
    #[inline]
    pub fn new(reduction_axes: Vec<isize>, output_axes: Vec<isize>) -> Self {
        Self { reduction_axes, output_axes }
    }

    /// Returns the axes whose extents multiply to the number of rows.
    #[inline]
    pub fn reduction_axes(&self) -> &[isize] {
        &self.reduction_axes
    }

    /// Returns the axes whose extents multiply to the number of columns.
    #[inline]
    pub fn output_axes(&self) -> &[isize] {
        &self.output_axes
    }

    /// Returns the batch, reduction, and output axes for a parameter of the shape `shape`, normalized to non-negative
    /// axes, with the batch axes in increasing order.
    #[allow(clippy::type_complexity)]
    pub(crate) fn resolve(&self, shape: &[usize]) -> Result<(Vec<usize>, Vec<usize>, Vec<usize>), OptimizerError> {
        let rank = shape.len();
        if rank < 2 {
            return Err(OptimizerError::InvalidConfiguration {
                message: format!("`ScaleByMuon` requires parameters of rank at least 2 but got rank {rank}"),
            });
        }
        let normalize =
            |axes: &[isize]| axes.iter().map(|axis| axis.rem_euclid(rank as isize) as usize).collect::<Vec<_>>();
        let reduction_axes = normalize(&self.reduction_axes);
        let output_axes = normalize(&self.output_axes);
        if reduction_axes.iter().any(|axis| output_axes.contains(axis)) {
            return Err(OptimizerError::InvalidConfiguration {
                message: format!(
                    "the reduction axes {reduction_axes:?} and the output axes {output_axes:?} of `ScaleByMuon` must \
                     be disjoint",
                ),
            });
        }
        let batch_axes = (0..rank)
            .filter(|axis| !reduction_axes.contains(axis) && !output_axes.contains(axis))
            .collect::<Vec<_>>();
        Ok((batch_axes, reduction_axes, output_axes))
    }
}

impl Default for MuonDimensionNumbers {
    fn default() -> Self {
        Self::new(vec![0], vec![1])
    }
}

/// Coefficients `(a, b, c)` of the Newton–Schulz iterations `X = a · X + (b · A + c · A²) · X` with `A = X Xᵀ` of
/// [`ScaleByMuon`], which are the `ns_coeffs` of [`optax.contrib.scale_by_muon`](
/// https://optax.readthedocs.io/en/latest/api/contrib.html#optax.contrib.scale_by_muon).
#[derive(Clone, Debug, PartialEq)]
pub enum NewtonSchulzCoefficients {
    /// One triple of coefficients for every iteration, of which there are as many as the number of Newton–Schulz steps.
    Constant([f64; 3]),

    /// One triple of coefficients per iteration, which must be at most as many as the number of Newton–Schulz steps
    /// and of which there are as many iterations as triples.
    PerStep(Vec<[f64; 3]>),
}

impl NewtonSchulzCoefficients {
    /// Returns the default coefficients `(3.4445, -4.7750, 2.0315)` of Muon, which is the `standard` preset of Optax.
    #[inline]
    pub fn standard() -> Self {
        Self::Constant([3.4445, -4.7750, 2.0315])
    }

    /// Returns the five triples of coefficients of [Dion](https://arxiv.org/abs/2504.05295), which is the `dion` preset
    /// of Optax.
    #[inline]
    pub fn dion() -> Self {
        Self::PerStep(vec![
            [4.0848, -6.8946, 2.9270],
            [3.9505, -6.3029, 2.6377],
            [3.7418, -5.5913, 2.3037],
            [2.8769, -3.1427, 1.2046],
            [2.8366, -3.0525, 1.2012],
        ])
    }
}

/// Preconditioning of the matrices that [`ScaleByMuon`] applies before its Newton–Schulz iterations, which is the
/// `preconditioning` of
/// [`optax.contrib.scale_by_muon`](https://optax.readthedocs.io/en/latest/api/contrib.html#optax.contrib.scale_by_muon)
/// without its `spectral` option, which needs the spectral norm of a matrix (i.e., a singular value decomposition)
/// that `ryft-core` does not provide.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MuonPreconditioning {
    /// Divides every matrix by its Frobenius norm plus `ε`.
    Frobenius,

    /// Rescales the rows of every matrix in the first iteration so that the absolute row sums of `X Xᵀ` are one, which
    /// is the almost orthogonal layer scaling of [Prach and Lampert (2022)](https://arxiv.org/abs/2208.03160).
    Aol,

    /// Divides every matrix in the first iteration by the square root of the Frobenius norm of `X Xᵀ` (i.e., by its
    /// Schatten 4-norm).
    Schatten,
}

parameterized_state! {
    /// State of [`ScaleByMuon`], which is the analogue of
    /// [`optax.contrib.MuonState`](https://optax.readthedocs.io/en/latest/api/contrib.html#optax.contrib.MuonState). It
    /// holds the step count, the momentum with the structure `S` of the parameters, and the Newton–Schulz coefficients
    /// (a vector of three coefficients or a matrix with one row of three coefficients per iteration, in `f32`).
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct ScaleByMuonState<P, S> with family ScaleByMuonStateFamily {
        /// Rank-0 `i32` array that holds the number of updates applied so far.
        count: P,

        /// Exponential moving averages of the gradients (i.e., the momentum), which are the `mu` of Optax.
        first_moments: S,

        /// Newton–Schulz coefficients, which are the `ns_coeffs` of Optax.
        newton_schulz_coefficients: P,
    }
}

impl<P: Parameter, S> ScaleByMuonState<P, S> {
    /// Returns the rank-0 `i32` array that holds the number of updates applied so far.
    #[inline]
    pub fn count(&self) -> &P {
        &self.count
    }

    /// Returns the exponential moving averages of the gradients.
    #[inline]
    pub fn first_moments(&self) -> &S {
        &self.first_moments
    }

    /// Returns the Newton–Schulz coefficients.
    #[inline]
    pub fn newton_schulz_coefficients(&self) -> &P {
        &self.newton_schulz_coefficients
    }
}

/// Transformation that orthogonalizes the bias corrected momentum of every parameter matrix with Newton–Schulz
/// iterations, which is the rescaling of the Muon optimizer of
/// [Jordan et al. (2024)](https://kellerjordan.github.io/posts/muon/) and the analogue of
/// [`optax.contrib.scale_by_muon`](
/// https://optax.readthedocs.io/en/latest/api/contrib.html#optax.contrib.scale_by_muon). For the gradients `gₜ` at the
/// step count `t` (i.e., the number of updates including the current one), it computes
///
/// ```text
/// μₜ = β · μₜ₋₁ + (1 − β) · gₜ
/// μ̂ₜ = β · μₜ / (1 − βᵗ⁺¹) + (1 − β) · gₜ / (1 − βᵗ)
/// X₀ = μ̂ₜ / (‖μ̂ₜ‖ + ε)
/// Aₖ = Xₖ₋₁ · Xₖ₋₁ᵀ
/// Xₖ = a · Xₖ₋₁ + (b · Aₖ + c · Aₖ²) · Xₖ₋₁   for k = 1, …, n
/// ```
///
/// and returns `Xₙ`. The momentum `μₜ` starts at zero and decays with the rate `β`, and `μ̂ₜ` is its Nesterov estimate.
/// Every matrix of `μ̂ₜ`, whose rows and columns are given by [`MuonDimensionNumbers`], is transposed first when it has
/// more rows than columns (and transposed back at the end), `‖·‖` is its Frobenius norm, `ε` keeps the division finite,
/// and the `n` Newton–Schulz iterations with the coefficients `(a, b, c)` drive its singular values towards one, so
/// `Xₙ` approximates the orthogonal factor of `μ̂ₜ`. The products in `Aₖ` and `Xₖ` are matrix products, and all other
/// operations are element-wise. The defaults are [`NewtonSchulzCoefficients::standard`] with `n = 5`, `β = 0.95`,
/// `ε = 1e-8`, Nesterov momentum, no adaptive scaling, and [`MuonPreconditioning::Frobenius`]. [`Self::with_nesterov`]
/// with `false` uses the bias corrected estimate `μ̂ₜ = μₜ / (1 − βᵗ)` instead, [`Self::with_adaptive`] returns
/// `⟨μ̂ₜ, Xₙ⟩ · Xₙ` with the sum `⟨·, ·⟩` of the element-wise products over the leaf, [`Self::with_preconditioning`]
/// replaces the division by the Frobenius norm with another [`MuonPreconditioning`] in the first iteration, and
/// [`Self::with_newton_schulz_coefficients`] sets the coefficients and the number of iterations.
///
/// Every leaf needs dimension numbers, which a function of its [`ParameterPath`] provides, or which default to
/// [`MuonDimensionNumbers::default`] for rank-2 leaves when there is no such function. Other leaves are rejected (refer
/// to [`Muon`](crate::Muon), which updates them with AdamW). Shapes must be static.
///
/// # References
///
///   - Jordan et al. (2024). [Muon: An optimizer for hidden layers in neural
///     networks](https://kellerjordan.github.io/posts/muon/).
///   - Bernstein and Newhouse (2024). [Old Optimizer, New Norm: An Anthology](https://arxiv.org/abs/2409.20325).
#[derive(Clone)]
pub struct ScaleByMuon {
    /// Coefficients of the Newton–Schulz iterations.
    newton_schulz_coefficients: NewtonSchulzCoefficients,

    /// Number of Newton–Schulz iterations with constant coefficients.
    newton_schulz_steps: usize,

    /// Decay `β` of the momentum.
    beta: f64,

    /// Term `ε` that is added to the Frobenius norm or that bounds the rescalings from below.
    epsilon: f64,

    /// Data type in which the momentum is stored, or [`None`] for the statistic data type of each leaf.
    first_moment_data_type: Option<DataType>,

    /// Whether the momentum is Nesterov momentum.
    nesterov: bool,

    /// Whether the orthogonalized updates are scaled by their inner products with the momentum.
    adaptive: bool,

    /// Preconditioning of the matrices before the iterations.
    preconditioning: MuonPreconditioning,

    /// Function that returns the dimension numbers of the leaf at a path, or [`None`] for the defaults of rank-2
    /// leaves.
    dimension_numbers: Option<Arc<dyn Fn(&ParameterPath) -> Option<MuonDimensionNumbers> + Send + Sync>>,
}

impl ScaleByMuon {
    /// Creates a new [`ScaleByMuon`] with default values for all hyperparameters.
    pub fn new() -> Self {
        Self {
            newton_schulz_coefficients: NewtonSchulzCoefficients::standard(),
            newton_schulz_steps: 5,
            beta: 0.95,
            epsilon: 1e-8,
            first_moment_data_type: None,
            nesterov: true,
            adaptive: false,
            preconditioning: MuonPreconditioning::Frobenius,
            dimension_numbers: None,
        }
    }

    /// Returns this transformation with the Newton–Schulz coefficients `coefficients` and `steps` steps.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when a coefficient is not finite, when there are no
    /// per-step coefficients, or when there are more per-step coefficients than steps.
    pub fn with_newton_schulz_coefficients(
        mut self,
        coefficients: NewtonSchulzCoefficients,
        steps: usize,
    ) -> Result<Self, OptimizerError> {
        let triples = match &coefficients {
            NewtonSchulzCoefficients::Constant(triple) => std::slice::from_ref(triple),
            NewtonSchulzCoefficients::PerStep(triples) => triples.as_slice(),
        };
        if triples.iter().flatten().any(|coefficient| !coefficient.is_finite()) {
            return Err(OptimizerError::InvalidConfiguration {
                message: "the Newton–Schulz coefficients must be finite".to_string(),
            });
        }
        if let NewtonSchulzCoefficients::PerStep(triples) = &coefficients
            && (triples.is_empty() || triples.len() > steps)
        {
            return Err(OptimizerError::InvalidConfiguration {
                message: format!(
                    "expected between 1 and {steps} triples of Newton–Schulz coefficients but got {}",
                    triples.len(),
                ),
            });
        }
        self.newton_schulz_coefficients = coefficients;
        self.newton_schulz_steps = steps;
        Ok(self)
    }

    /// Returns this transformation with `beta` as the decay `β` of its momentum.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `beta` is not in `[0, 1)`.
    pub fn with_beta(mut self, beta: f64) -> Result<Self, OptimizerError> {
        validate_decay_rate("beta", beta)?;
        self.beta = beta;
        Ok(self)
    }

    /// Returns this transformation with `epsilon` as its term `ε`.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `epsilon` is negative or not finite.
    pub fn with_epsilon(mut self, epsilon: f64) -> Result<Self, OptimizerError> {
        validate_non_negative("epsilon", epsilon)?;
        self.epsilon = epsilon;
        Ok(self)
    }

    /// Returns this transformation with its momentum stored in `data_type` (i.e., the `mu_dtype` of Optax).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `data_type` is not a floating-point data type.
    pub fn with_first_moment_data_type(mut self, data_type: DataType) -> Result<Self, OptimizerError> {
        validate_floating_point("first_moment_data_type", data_type)?;
        self.first_moment_data_type = Some(data_type);
        Ok(self)
    }

    /// Returns this transformation with Nesterov momentum when `nesterov` is `true`.
    #[inline]
    pub fn with_nesterov(mut self, nesterov: bool) -> Self {
        self.nesterov = nesterov;
        self
    }

    /// Returns this transformation with the adaptive scaling of its updates when `adaptive` is `true`.
    #[inline]
    pub fn with_adaptive(mut self, adaptive: bool) -> Self {
        self.adaptive = adaptive;
        self
    }

    /// Returns this transformation with `preconditioning` as the preconditioning of its matrices.
    #[inline]
    pub fn with_preconditioning(mut self, preconditioning: MuonPreconditioning) -> Self {
        self.preconditioning = preconditioning;
        self
    }

    /// Returns this transformation with `dimension_numbers` as the function that returns the dimension numbers of the
    /// leaf at a path (i.e., the `weight_dimension_numbers` of Optax). Leaves for which it returns [`None`] are
    /// rejected.
    pub fn with_dimension_numbers<DimensionNumbers>(mut self, dimension_numbers: DimensionNumbers) -> Self
    where
        DimensionNumbers: 'static + Fn(&ParameterPath) -> Option<MuonDimensionNumbers> + Send + Sync,
    {
        self.dimension_numbers = Some(Arc::new(dimension_numbers));
        self
    }

    /// Returns the Newton–Schulz coefficients of this transformation.
    #[inline]
    pub fn newton_schulz_coefficients(&self) -> &NewtonSchulzCoefficients {
        &self.newton_schulz_coefficients
    }

    /// Returns the number of Newton–Schulz steps of this transformation.
    #[inline]
    pub fn newton_schulz_steps(&self) -> usize {
        self.newton_schulz_steps
    }

    /// Returns the decay `β` of the momentum of this transformation.
    #[inline]
    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// Returns the term `ε` of this transformation.
    #[inline]
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// Returns the data type in which this transformation stores its momentum, or [`None`] if it stores it in the
    /// statistic data type of each leaf.
    #[inline]
    pub fn first_moment_data_type(&self) -> Option<DataType> {
        self.first_moment_data_type
    }

    /// Returns whether the momentum of this transformation is Nesterov momentum.
    #[inline]
    pub fn nesterov(&self) -> bool {
        self.nesterov
    }

    /// Returns whether this transformation scales its updates adaptively.
    #[inline]
    pub fn adaptive(&self) -> bool {
        self.adaptive
    }

    /// Returns the preconditioning of the matrices of this transformation.
    #[inline]
    pub fn preconditioning(&self) -> MuonPreconditioning {
        self.preconditioning
    }

    /// Returns the dimension numbers of the leaf at `path` of the rank `rank` for which [`Muon`](crate::Muon) selects
    /// it as a matrix leaf, which are those of the function of this transformation or, without one, the defaults for
    /// rank-2 leaves, and [`None`] for all other leaves.
    pub(crate) fn matrix_dimension_numbers(&self, path: &ParameterPath, rank: usize) -> Option<MuonDimensionNumbers> {
        match &self.dimension_numbers {
            Some(function) => function(path),
            None if rank == 2 => Some(MuonDimensionNumbers::default()),
            None => None,
        }
    }

    /// Returns the dimension numbers of the leaf at `path` with the shape `shape`, which are those of the function of
    /// this transformation or, without one, the defaults for rank-2 leaves.
    fn resolve_dimension_numbers(
        &self,
        path: &ParameterPath,
        shape: &[usize],
    ) -> Result<MuonDimensionNumbers, OptimizerError> {
        match &self.dimension_numbers {
            Some(dimension_numbers) => dimension_numbers(path).ok_or_else(|| OptimizerError::InvalidConfiguration {
                message: format!("`ScaleByMuon` has no dimension numbers for the leaf at `{path}`"),
            }),
            None if shape.len() == 2 => Ok(MuonDimensionNumbers::default()),
            None => Err(OptimizerError::InvalidConfiguration {
                message: format!(
                    "`ScaleByMuon` requires dimension numbers for the leaf at `{path}`, which has rank {}",
                    shape.len(),
                ),
            }),
        }
    }

    /// Returns the orthogonalization of `value` with the dimension numbers `dimension_numbers` and the coefficients
    /// `coefficients` (refer to [`ScaleByMuonState::newton_schulz_coefficients`]), which is
    /// `optax.contrib._muon.orthogonalize_via_newton_schulz` applied to every batch item.
    fn orthogonalize<P: OptimizerValue>(
        &self,
        value: &P,
        dimension_numbers: &MuonDimensionNumbers,
        coefficients: &P,
    ) -> Result<P, OptimizerError> {
        let shape = static_shape("ScaleByMuon", value)?;
        let (batch_axes, reduction_axes, output_axes) = dimension_numbers.resolve(&shape)?;
        let extents = |axes: &[usize]| axes.iter().map(|axis| shape[*axis]).collect::<Vec<_>>();
        let permutation = batch_axes.iter().chain(&reduction_axes).chain(&output_axes).copied().collect::<Vec<_>>();
        let unflattened_shape = permutation.iter().map(|axis| shape[*axis]).collect::<Vec<_>>();
        let rows = extents(&reduction_axes).iter().product::<usize>();
        let columns = extents(&output_axes).iter().product::<usize>();
        let batch = extents(&batch_axes).iter().product::<usize>();
        let mut matrices = value.transpose(permutation.clone())?.reshape([batch, rows, columns])?;
        let transposed = rows > columns;
        if transposed {
            matrices = transpose_matrices(&matrices)?;
        }
        let epsilon = matrices.fill_like(self.epsilon)?;
        if self.preconditioning == MuonPreconditioning::Frobenius {
            let norms = Add::add(&frobenius_norms(&matrices)?.broadcast_like(&matrices, &[0])?, &epsilon)?;
            matrices = Div::div(&matrices, &norms)?;
        }
        let iterations = match &self.newton_schulz_coefficients {
            NewtonSchulzCoefficients::Constant(_) => self.newton_schulz_steps,
            NewtonSchulzCoefficients::PerStep(triples) => triples.len(),
        };
        let coefficients = convert_to(coefficients, matrices.r#type().data_type())?;
        for iteration in 0..iterations {
            let coefficient = |index: usize| -> Result<P, OptimizerError> {
                let element = if coefficients.r#type().rank() == 1 {
                    coefficients.slice(&[index], &[index + 1], &[1])?
                } else {
                    coefficients.slice(&[iteration, index], &[iteration + 1, index + 1], &[1, 1])?
                };
                Ok(element.reshape(Shape::scalar())?)
            };
            let mut squared = gram_matrices(&matrices)?;
            if iteration == 0 && self.preconditioning == MuonPreconditioning::Aol {
                let rescaling = squared.abs()?.reduce(&[2], ReductionKind::Sum)?;
                let scale = rescaling.max(&rescaling.fill_like(self.epsilon)?)?.rsqrt()?;
                matrices = Mul::mul(&matrices, &scale.broadcast_like(&matrices, &[0, 1])?)?;
                squared = Mul::mul(
                    &Mul::mul(&squared, &scale.broadcast_like(&squared, &[0, 1])?)?,
                    &scale.broadcast_like(&squared, &[0, 2])?,
                )?;
            }
            if iteration == 0 && self.preconditioning == MuonPreconditioning::Schatten {
                let rescaling = Mul::mul(&squared, &squared)?.reduce(&[1, 2], ReductionKind::Sum)?.sqrt()?;
                let scale = rescaling.max(&rescaling.fill_like(self.epsilon)?)?.rsqrt()?;
                matrices = Mul::mul(&matrices, &scale.broadcast_like(&matrices, &[0])?)?;
                squared = Mul::mul(&squared, &Mul::mul(&scale, &scale)?.broadcast_like(&squared, &[0])?)?;
            }
            matrices =
                polynomial_iteration(&matrices, &squared, [&coefficient(0)?, &coefficient(1)?, &coefficient(2)?])?;
        }
        if transposed {
            matrices = transpose_matrices(&matrices)?;
        }
        let mut inverse_permutation = vec![0; permutation.len()];
        for (position, axis) in permutation.iter().enumerate() {
            inverse_permutation[*axis] = position;
        }
        Ok(matrices.reshape(unflattened_shape)?.transpose(inverse_permutation)?)
    }

    /// Returns the [`ScaleByMuonState::newton_schulz_coefficients`] of this transformation, in `f32` and in the domain
    /// of `count`.
    fn coefficients<P: OptimizerValue>(&self, count: &P) -> Result<P, OptimizerError> {
        let scalar = convert_to(count, DataType::F32)?;
        let triples = match &self.newton_schulz_coefficients {
            NewtonSchulzCoefficients::Constant(triple) => std::slice::from_ref(triple),
            NewtonSchulzCoefficients::PerStep(triples) => triples.as_slice(),
        };
        let elements = triples
            .iter()
            .flatten()
            .map(|coefficient| Ok(scalar.fill_like(*coefficient)?.reshape([1])?))
            .collect::<Result<Vec<_>, OptimizerError>>()?;
        let vector = Concatenate::concatenate(&elements, 0)?;
        Ok(match &self.newton_schulz_coefficients {
            NewtonSchulzCoefficients::Constant(_) => vector,
            NewtonSchulzCoefficients::PerStep(triples) => vector.reshape([triples.len(), 3])?,
        })
    }
}

impl Default for ScaleByMuon {
    fn default() -> Self {
        Self::new()
    }
}

impl Debug for ScaleByMuon {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The dimension numbers function is opaque, so only its presence is rendered.
        formatter
            .debug_struct("ScaleByMuon")
            .field("newton_schulz_coefficients", &self.newton_schulz_coefficients)
            .field("newton_schulz_steps", &self.newton_schulz_steps)
            .field("beta", &self.beta)
            .field("epsilon", &self.epsilon)
            .field("first_moment_data_type", &self.first_moment_data_type)
            .field("nesterov", &self.nesterov)
            .field("adaptive", &self.adaptive)
            .field("preconditioning", &self.preconditioning)
            .field("dimension_numbers", &self.dimension_numbers.is_some())
            .finish()
    }
}

impl<P: OptimizerValue> Optimizer<P> for ScaleByMuon {
    type State<S: Parameterized<P>> = ScaleByMuonState<P, S>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        let count = initial_count("ScaleByMuon", parameters)?;
        Ok(ScaleByMuonState {
            newton_schulz_coefficients: self.coefficients(&count)?,
            count,
            first_moments: filled_statistics(parameters, 0.0, self.first_moment_data_type)?,
        })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        // This follows the operation order of `optax.contrib.scale_by_muon`.
        let _ = arguments;
        let paths = gradients.parameter_paths().collect::<Vec<_>>();
        validate_structure("parameters", &paths, parameters)?;
        validate_structure("first_moments", &paths, &state.first_moments)?;
        let count = increment_count(&state.count)?;
        let step = convert_to(&count, DataType::F32)?;
        let correction = bias_correction(&step, self.beta)?;
        let next_correction = bias_correction(&convert_to(&increment_count(&count)?, DataType::F32)?, self.beta)?;
        let structure = gradients.parameter_structure();
        let mut updates = Vec::with_capacity(paths.len());
        let mut moments = Vec::with_capacity(paths.len());
        for ((gradient, moment), path) in
            gradients.into_parameters().zip(state.first_moments.into_parameters()).zip(&paths)
        {
            let data_type = gradient.r#type().data_type();
            let statistic_type = statistic_data_type(data_type);
            let moment_type = moment.r#type().data_type();
            let gradient = convert_to(&gradient, statistic_type)?;
            let moment = update_moment(&gradient, &convert_to(&moment, statistic_type)?, self.beta)?;
            let corrected = if self.nesterov {
                Add::add(
                    &Mul::mul(&gradient.fill_like(self.beta)?, &bias_corrected(&moment, &next_correction)?)?,
                    &Mul::mul(&gradient.fill_like(1.0 - self.beta)?, &bias_corrected(&gradient, &correction)?)?,
                )?
            } else {
                bias_corrected(&moment, &correction)?
            };
            let dimension_numbers = self.resolve_dimension_numbers(path, &static_shape("ScaleByMuon", &gradient)?)?;
            let mut update = self.orthogonalize(&corrected, &dimension_numbers, &state.newton_schulz_coefficients)?;
            if self.adaptive {
                let scale = Mul::mul(&corrected, &update)?.reduce(&all_axes(&update), ReductionKind::Sum)?;
                update = Mul::mul(&scale.broadcast_like(&update, &[])?, &update)?;
            }
            updates.push(convert_to(&update, data_type)?);
            moments.push(convert_to(&moment, moment_type)?);
        }
        let state = ScaleByMuonState {
            count,
            first_moments: S::from_parameters(structure.clone(), moments)?,
            newton_schulz_coefficients: state.newton_schulz_coefficients,
        };
        Ok((S::from_parameters(structure, updates)?, state))
    }
}

parameterized_state! {
    /// State of [`ScaleByDimuon`], which holds the momentum with the structure `S` of the parameters. It holds no step
    /// count, because the update of the reconstruction does not depend on the number of steps.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct ScaleByDimuonState<P, S> with family ScaleByDimuonStateFamily and marker leaf {
        /// Exponential moving averages of the gradients (i.e., the momentum), which start at zero.
        first_moments: S,
    }
}

impl<P: Parameter, S> ScaleByDimuonState<P, S> {
    /// Returns the exponential moving averages of the gradients (i.e., the momentum).
    #[inline]
    pub fn first_moments(&self) -> &S {
        &self.first_moments
    }
}

/// Transformation that orthogonalizes the row and column conditioned momentum of every LoRA factor, which is the
/// rescaling of the Dimuon optimizer of Thinking Machines. Its official formulation is unpublished, so this is the
/// unofficial reconstruction of [anpaure/dimuon](https://github.com/anpaure/dimuon), which may differ from the
/// optimizer that the Tinker API serves. For every matrix `W` of shape `m × n` of a leaf, with the gradient `gₜ` and
/// the momentum `Mₜ`, it computes
///
/// ```text
/// eʳᵢ = Σⱼ (Mₜ₋₁)ᵢⱼ²
/// eᶜⱼ = Σᵢ (Mₜ₋₁)ᵢⱼ²
/// L   = diag((eʳ / max eʳ + δ)^(−1/4))
/// R   = diag((eᶜ / max eᶜ + δ)^(−1/4))
/// Mₜ  = β₁ · Mₜ₋₁ + (1 − β₁) · gₜ
/// Pₜ  = polar(L · Mₜ · R)
/// Dₜ  = L · Pₜ · R
/// uₜ  = √r · (r / 32)^p · Dₜ / ‖Dₜ‖
/// ```
///
/// and returns `uₜ`, which is the positive direction without the learning rate and its sign. The row energies `eʳ` and
/// the column energies `eᶜ` are those of the previous momentum `Mₜ₋₁`, which starts at zero, so the diagonal
/// conditioners `L` and `R` are the identity on the first step (and whenever `max eʳ` or `max eᶜ` is zero). `δ` is the
/// relative damping of the energies, `β₁` is the decay of the momentum, `r = min(m, n)` is the rank of the matrix, `p`
/// is the exponent of its rank scaling relative to the reference rank `32`, and `‖·‖` is the Frobenius norm. The
/// update is zero when `L · Mₜ · R` is zero. `polar` is five Polar Express iterations, which drive the singular values
/// of a matrix towards one. A matrix with more rows than columns is transposed first (and transposed back at the end),
/// and is then normalized as `X₀ = X / (1.01 · ‖X‖)` and iterated as `Xₖ = a · Xₖ₋₁ + (b · Gₖ + c · Gₖ²) · Xₖ₋₁` with
/// `Gₖ = Xₖ₋₁ · Xₖ₋₁ᵀ`, for the five coefficient triples `(a, b, c)` of the reconstruction rescaled as
/// `(a / 1.01, b / 1.01³, c / 1.01⁵)`. The products of `L`, `R`, `Mₜ`, `Pₜ`, and `Xₖ` are matrix products, and all
/// other operations are element-wise. Every update therefore has the Frobenius norm `√r · (r / 32)^p`. The defaults
/// are `β₁ = 0.9`, `δ = 1e-4`, and `p = 0.175`. [`Self::with_beta1`] and [`Self::with_damping`] set `β₁` and `δ`, and
/// [`Self::with_rank_scaling_exponent`] sets `p` or, with [`None`], removes the factor `(r / 32)^p`, so that the
/// updates have the Frobenius norm `√r`. A zero damping makes the conditioners infinite for rows or columns without
/// energy (i.e., NaN updates), as in the reconstruction. There is no second moment, no bias correction, and no weight
/// decay.
///
/// A leaf of shape `[k₁, …, kₗ, m, n]` is a batch of `k₁ · … · kₗ` matrices of shape `m × n` (e.g., the
/// `[experts, m, n]` factors of mixtures of experts), and the conditioners, the polar iterations, and the normalization
/// apply to every matrix separately. Leaves of rank 0 or 1 are not LoRA factors and are rejected, so they should be
/// routed to another optimizer with [`Partition`](crate::Partition) or [`Masked`](crate::Masked). Shapes must be
/// static. Unlike the reconstruction, which stores the momentum in `bf16` and runs the polar iterations in `fp16`, this
/// transformation computes everything in `f32` (or in `f64` for `f64` gradients) and stores the momentum in the
/// statistic data type of every parameter, like the other transformations of this crate.
///
/// # References
///
///   - Thinking Machines. [Tinker documentation:
///     optimizers](https://tinker-docs.thinkingmachines.ai/tinker/optimizers/), which lists the hyperparameters of
///     Dimuon but not its formulation.
///   - anpaure. [Dimuon reverse engineered](https://github.com/anpaure/dimuon), the reconstruction implemented here.
///   - catid. [Reproduction of the Dimuon optimizer](https://github.com/catid/dimuon).
///   - Jordan et al. (2024). [Muon: An optimizer for hidden layers in neural
///     networks](https://kellerjordan.github.io/posts/muon/).
///   - Amsel, Persson, Musco, and Gower (2025). [The Polar Express: Optimal Matrix Sign Methods and Their Application
///     to the Muon Algorithm](https://arxiv.org/abs/2505.16932).
#[derive(Clone, Debug, PartialEq)]
pub struct ScaleByDimuon {
    /// Decay `β₁` of the momentum.
    beta1: f64,

    /// Relative damping `δ` of the row and column energies.
    damping: f64,

    /// Exponent `p` of the rank scaling relative to the reference rank, or [`None`] without that factor.
    rank_scaling_exponent: Option<f64>,
}

impl ScaleByDimuon {
    /// Creates a new [`ScaleByDimuon`] with default values for all hyperparameters.
    pub fn new() -> Self {
        Self { beta1: 0.9, damping: 1e-4, rank_scaling_exponent: Some(0.175) }
    }

    /// Returns this transformation with `beta1` as the decay `β₁` of its momentum.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `beta1` is not in `[0, 1)`.
    pub fn with_beta1(mut self, beta1: f64) -> Result<Self, OptimizerError> {
        validate_decay_rate("beta1", beta1)?;
        self.beta1 = beta1;
        Ok(self)
    }

    /// Returns this transformation with `damping` as the relative damping `δ` of its row and column energies.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `damping` is negative or not finite.
    pub fn with_damping(mut self, damping: f64) -> Result<Self, OptimizerError> {
        validate_non_negative("damping", damping)?;
        self.damping = damping;
        Ok(self)
    }

    /// Returns this transformation with `exponent` as the exponent `p` of the rank scaling `(r / 32)^p` of its
    /// updates, or without that factor (i.e., with updates of the Frobenius norm `√r`) when `exponent` is [`None`].
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `exponent` is not finite.
    pub fn with_rank_scaling_exponent(mut self, exponent: Option<f64>) -> Result<Self, OptimizerError> {
        if let Some(exponent) = exponent {
            validate_finite("rank_scaling_exponent", exponent)?;
        }
        self.rank_scaling_exponent = exponent;
        Ok(self)
    }

    /// Returns the decay `β₁` of the momentum of this transformation.
    #[inline]
    pub fn beta1(&self) -> f64 {
        self.beta1
    }

    /// Returns the relative damping `δ` of the row and column energies of this transformation.
    #[inline]
    pub fn damping(&self) -> f64 {
        self.damping
    }

    /// Returns the exponent `p` of the rank scaling of this transformation, if it scales its updates by `(r / 32)^p`.
    #[inline]
    pub fn rank_scaling_exponent(&self) -> Option<f64> {
        self.rank_scaling_exponent
    }

    /// Returns the update of the leaf at `path` and its next momentum for its `gradient` and its previous momentum
    /// `momentum`, which both have the statistic data type of the gradient.
    fn update_leaf<P: OptimizerValue>(
        &self,
        path: &ParameterPath,
        gradient: &P,
        momentum: &P,
    ) -> Result<(P, P), OptimizerError> {
        let shape = static_shape("ScaleByDimuon", gradient)?;
        let (batch, rows, columns) = matrix_extents(path, &shape)?;
        let next_momentum = update_moment(gradient, momentum, self.beta1)?;
        let previous = momentum.reshape([batch, rows, columns])?;
        let matrices = next_momentum.reshape([batch, rows, columns])?;
        let squared = Mul::mul(&previous, &previous)?;
        let left = self.conditioner(&squared.reduce(&[2], ReductionKind::Sum)?)?.broadcast_like(&matrices, &[0, 1])?;
        let right = self.conditioner(&squared.reduce(&[1], ReductionKind::Sum)?)?.broadcast_like(&matrices, &[0, 2])?;
        let conditioned = Mul::mul(&Mul::mul(&left, &matrices)?, &right)?;
        let direction = Mul::mul(&Mul::mul(&left, &polar(&conditioned, rows > columns)?)?, &right)?;
        let norms = frobenius_norms(&direction)?;
        let norms = norms.max(&norms.fill_like(smallest_normal(norms.r#type().data_type()))?)?;
        let rank = rows.min(columns) as f64;
        let scale = rank.sqrt() * self.rank_scaling_exponent.map_or(1.0, |exponent| (rank / 32.0).powf(exponent));
        let update =
            Mul::mul(&direction.fill_like(scale)?, &Div::div(&direction, &norms.broadcast_like(&direction, &[0])?)?)?;
        Ok((update.reshape(shape)?, next_momentum))
    }

    /// Returns the diagonals of the conditioners `(e / max e + δ)^(−1/4)` for the energies `energies` of shape
    /// `[batch, extent]`, or ones for the matrices whose energies are all zero.
    fn conditioner<P: OptimizerValue>(&self, energies: &P) -> Result<P, OptimizerError> {
        let peaks = energies.reduce(&[1], ReductionKind::Max)?;
        let denominators = peaks.max(&peaks.fill_like(smallest_normal(peaks.r#type().data_type()))?)?;
        let damped = Add::add(
            &Div::div(energies, &denominators.broadcast_like(energies, &[0])?)?,
            &energies.fill_like(self.damping)?,
        )?;
        let conditioners = damped.sqrt()?.rsqrt()?;
        let live = peaks.compare(&peaks.fill_like(0f64)?, ComparisonDirection::GreaterThan)?;
        Ok(live.broadcast_like(energies, &[0])?.select(&conditioners, &energies.fill_like(1f64)?)?)
    }
}

impl Default for ScaleByDimuon {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: OptimizerValue> Optimizer<P> for ScaleByDimuon {
    type State<S: Parameterized<P>> = ScaleByDimuonState<P, S>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        for (path, parameter) in parameters.named_parameters() {
            matrix_extents(&path, &static_shape("ScaleByDimuon", parameter)?)?;
        }
        Ok(ScaleByDimuonState { first_moments: filled_statistics(parameters, 0.0, None)?, leaf: PhantomData })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        // This follows `dimuon.py` of the reconstruction, with the statistics computed in the statistic data type of
        // every gradient and every new momentum stored in the data type of the one that it replaces.
        let _ = arguments;
        let paths = gradients.parameter_paths().collect::<Vec<_>>();
        validate_structure("parameters", &paths, parameters)?;
        validate_structure("first_moments", &paths, &state.first_moments)?;
        let structure = gradients.parameter_structure();
        let mut updates = Vec::with_capacity(paths.len());
        let mut moments = Vec::with_capacity(paths.len());
        for ((gradient, moment), path) in
            gradients.into_parameters().zip(state.first_moments.into_parameters()).zip(&paths)
        {
            let data_type = gradient.r#type().data_type();
            let statistic_type = statistic_data_type(data_type);
            let moment_type = moment.r#type().data_type();
            let (update, moment) =
                self.update_leaf(path, &convert_to(&gradient, statistic_type)?, &convert_to(&moment, statistic_type)?)?;
            updates.push(convert_to(&update, data_type)?);
            moments.push(convert_to(&moment, moment_type)?);
        }
        let state =
            ScaleByDimuonState { first_moments: S::from_parameters(structure.clone(), moments)?, leaf: PhantomData };
        Ok((S::from_parameters(structure, updates)?, state))
    }
}

/// Raw coefficient triples `(a, b, c)` of the five Polar Express iterations of the reconstruction, which [`polar`]
/// rescales as `(a / 1.01, b / 1.01³, c / 1.01⁵)` so that the iterations stay stable for singular values up to `1.01`.
const POLAR_EXPRESS_COEFFICIENTS: [[f64; 3]; 5] = [
    [8.28721201814563, -23.595886519098837, 17.300387312530933],
    [4.107059111542203, -2.9478499167379106, 0.5448431082926601],
    [3.9486908534822946, -2.908902115962949, 0.5518191394370137],
    [3.3184196573706015, -2.488488024314874, 0.51004894012372],
    [2.300652019954817, -1.6689039845747493, 0.4188073119525673],
];

/// Returns the number of matrices, the number of rows, and the number of columns of the leaf at `path` of the shape
/// `shape`, whose last two dimensions are the rows and the columns of its matrices and whose leading dimensions are
/// batch dimensions.
fn matrix_extents(path: &ParameterPath, shape: &[usize]) -> Result<(usize, usize, usize), OptimizerError> {
    let rank = shape.len();
    if rank < 2 {
        return Err(OptimizerError::InvalidConfiguration {
            message: format!(
                "`ScaleByDimuon` requires leaves of rank at least 2 but the leaf at `{path}` has rank {rank}, so it \
                 should be routed to another optimizer with `Partition` or `Masked`",
            ),
        });
    }
    Ok((shape[..rank - 2].iter().product(), shape[rank - 2], shape[rank - 1]))
}

/// Returns the smallest positive normal number of `data_type`, which is `f64::MIN_POSITIVE` for `f64` and
/// `f32::MIN_POSITIVE` for all other data types, and which bounds norms and energies from below so that zero matrices
/// are divided by a positive number.
fn smallest_normal(data_type: DataType) -> f64 {
    if data_type == DataType::F64 { f64::MIN_POSITIVE } else { f32::MIN_POSITIVE as f64 }
}

/// Returns the approximate orthogonal factors of the matrices of the batch `matrices` of shape `[batch, rows, columns]`
/// after five Polar Express iterations, where `transposed` states that the matrices have more rows than columns and are
/// therefore iterated as their transposes. Zero matrices give zero factors.
fn polar<P: OptimizerValue>(matrices: &P, transposed: bool) -> Result<P, OptimizerError> {
    let mut matrices = if transposed { transpose_matrices(matrices)? } else { matrices.clone() };
    let data_type = matrices.r#type().data_type();
    let norms = frobenius_norms(&matrices)?;
    let denominators = Mul::mul(&norms.fill_like(1.01)?, &norms)?.max(&norms.fill_like(smallest_normal(data_type))?)?;
    matrices = Div::div(&matrices, &denominators.broadcast_like(&matrices, &[0])?)?;
    let scalar = convert_to(&zero_count_like(&matrices)?, data_type)?;
    for [a, b, c] in POLAR_EXPRESS_COEFFICIENTS {
        let coefficients = [
            scalar.fill_like(a / 1.01)?,
            scalar.fill_like(b / 1.01f64.powi(3))?,
            scalar.fill_like(c / 1.01f64.powi(5))?,
        ];
        let gram = gram_matrices(&matrices)?;
        matrices = polynomial_iteration(&matrices, &gram, [&coefficients[0], &coefficients[1], &coefficients[2]])?;
    }
    if transposed { transpose_matrices(&matrices) } else { Ok(matrices) }
}

/// Returns the static extents of the shape of `value`, or an error naming `optimizer` when its shape is dynamic, which
/// the matrix optimizers of Muon and Dimuon need to arrange the leaves as batches of matrices.
pub(crate) fn static_shape<P: OptimizerValue>(optimizer: &str, value: &P) -> Result<Vec<usize>, OptimizerError> {
    let r#type = value.r#type();
    r#type
        .shape()
        .dimensions()
        .iter()
        .map(|dimension: &Dimension| {
            dimension.value().ok_or_else(|| OptimizerError::InvalidConfiguration {
                message: format!("`{optimizer}` requires static shapes but got `{}`", r#type),
            })
        })
        .collect()
}

/// Returns the batch of matrices `matrices` of shape `[batch, rows, columns]` with the rows and the columns of every
/// matrix swapped (i.e., the batch of the transposed matrices, of shape `[batch, columns, rows]`).
fn transpose_matrices<P: OptimizerValue>(matrices: &P) -> Result<P, OptimizerError> {
    Ok(matrices.transpose([0, 2, 1])?)
}

/// Returns the rank-1 array of the Frobenius norms (i.e., the square roots of the sums of the squared elements) of the
/// matrices of the batch `matrices` of shape `[batch, rows, columns]`.
fn frobenius_norms<P: OptimizerValue>(matrices: &P) -> Result<P, OptimizerError> {
    Ok(Mul::mul(matrices, matrices)?.reduce(&[1, 2], ReductionKind::Sum)?.sqrt()?)
}

/// Returns the products `left · right` of the matrices at the same positions of the batches `left` of shape
/// `[batch, rows, inner]` and `right` of shape `[batch, inner, columns]`.
fn matrix_products<P: OptimizerValue>(left: &P, right: &P) -> Result<P, OptimizerError> {
    Ok(left.dot(right, &DotDimensionNumbers::new(vec![2], vec![1], vec![0], vec![0]))?)
}

/// Returns the Gram matrices `X · Xᵀ` of the matrices `X` of the batch `matrices` of shape `[batch, rows, columns]`.
fn gram_matrices<P: OptimizerValue>(matrices: &P) -> Result<P, OptimizerError> {
    Ok(matrices.dot(matrices, &DotDimensionNumbers::new(vec![2], vec![2], vec![0], vec![0]))?)
}

/// Returns one polynomial iteration `a · X + (b · G + c · G · G) · X` of the matrices `X` of the batch `matrices`,
/// whose Gram matrices `G` (refer to [`gram_matrices`], possibly rescaled together with `X`) are `gram`, for the rank-0
/// `coefficients` `[a, b, c]` of the data type of `matrices`. The products of matrices are batched matrix products and
/// all other operations are element-wise. It maps every singular value `σ` of `X` to `a · σ + b · σ³ + c · σ⁵`, which
/// is how both the Newton–Schulz iterations of [`ScaleByMuon`] and the Polar Express iterations of [`ScaleByDimuon`]
/// approximate the orthogonal factor of a matrix.
fn polynomial_iteration<P: OptimizerValue>(matrices: &P, gram: &P, coefficients: [&P; 3]) -> Result<P, OptimizerError> {
    let [a, b, c] = coefficients;
    let polynomial = Add::add(
        &Mul::mul(&b.broadcast_like(gram, &[])?, gram)?,
        &Mul::mul(&c.broadcast_like(gram, &[])?, &matrix_products(gram, gram)?)?,
    )?;
    Ok(Add::add(&Mul::mul(&a.broadcast_like(matrices, &[])?, matrices)?, &matrix_products(&polynomial, matrices)?)?)
}

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::BTreeMap;

    use pretty_assertions::assert_eq;

    use ryft_core::{Array, ArrayType, Shape, Typed};

    use crate::tests::{assert_parity_with_tolerance, state_paths};

    use super::*;

    /// Returns the singular value that the Newton–Schulz iterations with the coefficients `coefficients` map the
    /// singular value `value` to, in `f64`. The iterations act on every singular value separately (i.e.,
    /// `σ = a · σ + b · σ³ + c · σ⁵`), which is how the tests derive the expected orthogonalizations.
    pub(crate) fn iterate(value: f64, coefficients: &[[f64; 3]]) -> f64 {
        coefficients
            .iter()
            .fold(value, |value, [a, b, c]| a * value + b * value.powi(3) + c * value.powi(5))
    }

    /// Returns the standard coefficients repeated for five steps.
    pub(crate) fn standard() -> Vec<[f64; 3]> {
        vec![[3.4445, -4.7750, 2.0315]; 5]
    }

    /// Asserts that the `f32` elements of `actual` are within `1e-4` of `expected`.
    pub(crate) fn assert_close(actual: &Array, expected: &[f64]) {
        let actual = actual.elements::<f32>().unwrap();
        assert_eq!(actual.len(), expected.len());
        assert!(
            actual.iter().zip(expected).all(|(actual, expected)| (*actual as f64 - expected).abs() < 1e-4),
            "{actual:?} != {expected:?}",
        );
    }

    /// Tolerance of the comparisons of `f32` updates with the `float64` reference script, relative to the Frobenius
    /// norm of the update of each leaf (refer to [`assert_reference`]). The first polar iterations multiply small
    /// singular values by about `8`, `4`, `4`, `3`, and `2`, so the rounding errors of `f32` arithmetic (about `6e-8`)
    /// grow to about `1e-4` of the norm of an update. The largest error of the reference tests is `8.8e-5` of that
    /// norm, and the same updates computed in `f64` match the reference within `1e-6`, which is the rounding of the
    /// `f32` constants.
    pub(crate) const REFERENCE_TOLERANCE: f64 = 2e-4;

    /// Returns the reference leaf `leaf` of the data type `data_type` with the elements `elements`, which is the wide
    /// 3 × 4 matrix for leaf `0`, the tall 4 × 3 matrix for leaf `1`, and the batch of two 2 × 3 matrices of shape
    /// `[2, 2, 3]` for leaf `2`.
    pub(crate) fn reference_leaf(leaf: usize, data_type: DataType, elements: &[f32]) -> Array {
        let shape = match leaf {
            0 => Shape::from([3, 4]),
            1 => Shape::from([4, 3]),
            _ => Shape::from([2, 2, 3]),
        };
        let r#type = ArrayType::new(data_type, shape);
        if data_type == DataType::F64 {
            Array::from_elements(r#type, &elements.iter().map(|element| *element as f64).collect::<Vec<_>>()).unwrap()
        } else {
            Array::from_elements(r#type, elements).unwrap()
        }
    }

    /// Returns the gradients of the reference leaves of the data type `data_type` at the step `step` (counting from
    /// `0`), whose element `i` of leaf `k` is `f32(sₜ · sin(0.7 · i + 1.3 · t + 0.5 · k + 0.1))` with
    /// `s = [1.0, 0.8, 0.05, 1.5]`, which is how the reference script `dimuon_reference.py` creates them.
    fn reference_gradients(step: usize, data_type: DataType) -> Vec<Array> {
        let scale = [1.0, 0.8, 0.05, 1.5][step];
        (0..3)
            .map(|leaf| {
                let elements = (0..12)
                    .map(|index| {
                        let angle = 0.7 * index as f64 + 1.3 * step as f64 + 0.5 * leaf as f64 + 0.1;
                        (scale * angle.sin()) as f32
                    })
                    .collect::<Vec<_>>();
                reference_leaf(leaf, data_type, &elements)
            })
            .collect()
    }

    /// Returns the elements of the updates of every reference leaf of the data type `data_type` that `optimizer`
    /// computes for the four reference steps, starting from its initial state for zero parameters, rounded to `f32`.
    pub(crate) fn reference_updates<O: Optimizer<Array>>(optimizer: &O, data_type: DataType) -> Vec<Vec<Vec<f32>>> {
        let parameters = (0..3).map(|leaf| reference_leaf(leaf, data_type, &[0.0; 12])).collect::<Vec<_>>();
        let mut state = optimizer.initialize(&parameters).unwrap();
        let mut updates = Vec::new();
        for step in 0..4 {
            let gradients = reference_gradients(step, data_type);
            let (step_updates, next_state) =
                optimizer.update(gradients, state, &parameters, &UpdateArguments::new()).unwrap();
            updates.push(
                step_updates
                    .iter()
                    .map(|update| match data_type {
                        DataType::F64 => {
                            update.elements::<f64>().unwrap().iter().map(|element| *element as f32).collect()
                        }
                        _ => update.elements::<f32>().unwrap(),
                    })
                    .collect(),
            );
            state = next_state;
        }
        updates
    }

    /// Asserts that the updates `actual` of the reference leaves at every step match the updates `expected` of the
    /// reference script, element by element, within `tolerance` times the Frobenius norm of the expected update of
    /// the same leaf. The polar iterations amplify rounding errors by their derivatives, which are largest for small
    /// singular values, so the errors of `f32` updates scale with the norm of the update rather than with the
    /// magnitude of each element.
    #[track_caller]
    pub(crate) fn assert_reference(actual: &[Vec<Vec<f32>>], expected: &[[[f32; 12]; 3]; 4], tolerance: f64) {
        assert_eq!(actual.len(), expected.len());
        for (step, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            for (leaf, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                let bound = tolerance * matrix_norms(expected, 1)[0];
                let error = actual
                    .iter()
                    .zip(expected)
                    .map(|(actual, expected)| (*actual as f64 - *expected as f64).abs())
                    .fold(0.0, f64::max);
                assert!(
                    error <= bound,
                    "leaf {leaf} at step {step} differs from the reference\n  actual:   {actual:?}\n  expected: \
                     {expected:?}",
                );
            }
        }
    }

    /// Returns the Frobenius norms of the `matrices` consecutive matrices of the same size that the `f32` elements
    /// `elements` hold, in `f64`.
    pub(crate) fn matrix_norms(elements: &[f32], matrices: usize) -> Vec<f64> {
        elements
            .chunks(elements.len() / matrices)
            .map(|matrix| matrix.iter().map(|element| (*element as f64).powi(2)).sum::<f64>().sqrt())
            .collect()
    }

    /// Returns the singular value that the five rescaled Polar Express iterations map the singular value `value` to,
    /// in `f64`. The iterations act on every singular value separately (i.e., `σ = a · σ + b · σ³ + c · σ⁵`).
    fn polar_express(value: f64) -> f64 {
        POLAR_EXPRESS_COEFFICIENTS.iter().fold(value, |value, [a, b, c]| {
            a / 1.01 * value + b / 1.01f64.powi(3) * value.powi(3) + c / 1.01f64.powi(5) * value.powi(5)
        })
    }

    #[test]
    fn test_muon_dimension_numbers() {
        let numbers = MuonDimensionNumbers::new(vec![-2], vec![-1]);
        assert_eq!((numbers.reduction_axes(), numbers.output_axes()), (&[-2isize][..], &[-1isize][..]));
        assert_eq!(numbers.resolve(&[4, 2, 3]).unwrap(), (vec![0], vec![1], vec![2]));
        assert_eq!(MuonDimensionNumbers::default(), MuonDimensionNumbers::new(vec![0], vec![1]));
        assert!(matches!(
            MuonDimensionNumbers::new(vec![0], vec![-2]).resolve(&[2, 3]),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "the reduction axes [0] and the output axes [0] of `ScaleByMuon` must be disjoint",
        ));
        assert!(matches!(
            MuonDimensionNumbers::default().resolve(&[2]),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`ScaleByMuon` requires parameters of rank at least 2 but got rank 1",
        ));
    }

    #[test]
    fn test_scale_by_muon_new() {
        let muon = ScaleByMuon::new();
        assert_eq!(muon.newton_schulz_coefficients(), &NewtonSchulzCoefficients::standard());
        assert_eq!(muon.newton_schulz_steps(), 5);
        assert_eq!((muon.beta(), muon.epsilon()), (0.95, 1e-8));
        assert_eq!(muon.first_moment_data_type(), None);
        assert!(muon.nesterov());
        assert!(!muon.adaptive());
        assert_eq!(muon.preconditioning(), MuonPreconditioning::Frobenius);
        assert!(format!("{muon:?}").ends_with("preconditioning: Frobenius, dimension_numbers: false }"));
    }

    #[test]
    fn test_scale_by_muon_with_newton_schulz_coefficients() {
        let muon = ScaleByMuon::new().with_newton_schulz_coefficients(NewtonSchulzCoefficients::dion(), 5).unwrap();
        assert_eq!(muon.newton_schulz_coefficients(), &NewtonSchulzCoefficients::dion());
        assert!(matches!(
            ScaleByMuon::new().with_newton_schulz_coefficients(NewtonSchulzCoefficients::dion(), 4),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "expected between 1 and 4 triples of Newton–Schulz coefficients but got 5",
        ));
        assert!(matches!(
            ScaleByMuon::new()
                .with_newton_schulz_coefficients(NewtonSchulzCoefficients::Constant([f64::NAN, 0.0, 0.0]), 5),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "the Newton–Schulz coefficients must be finite",
        ));

        // The per-step coefficients are applied once each.
        let parameters = vec![Array::matrix(2, 2, vec![0.0f32; 4]).unwrap()];
        let state = muon.initialize(&parameters).unwrap();
        assert_eq!(state.newton_schulz_coefficients().r#type().shape(), &Shape::from([5, 3]));
        let gradients = vec![Array::matrix(2, 2, vec![3.0f32, 0.0, 0.0, 4.0]).unwrap()];
        let (updates, _) = muon.update(gradients, state, &parameters, &UpdateArguments::new()).unwrap();
        let dion = [
            [4.0848, -6.8946, 2.9270],
            [3.9505, -6.3029, 2.6377],
            [3.7418, -5.5913, 2.3037],
            [2.8769, -3.1427, 1.2046],
            [2.8366, -3.0525, 1.2012],
        ];
        assert_close(&updates[0], &[iterate(0.6, &dion), 0.0, 0.0, iterate(0.8, &dion)]);
    }

    #[test]
    fn test_scale_by_muon_with_hyperparameters() {
        let muon = ScaleByMuon::new()
            .with_beta(0.9)
            .unwrap()
            .with_epsilon(1e-6)
            .unwrap()
            .with_first_moment_data_type(DataType::BF16)
            .unwrap()
            .with_nesterov(false)
            .with_adaptive(true)
            .with_preconditioning(MuonPreconditioning::Aol)
            .with_dimension_numbers(|_| Some(MuonDimensionNumbers::default()));
        assert_eq!((muon.beta(), muon.epsilon()), (0.9, 1e-6));
        assert_eq!(muon.first_moment_data_type(), Some(DataType::BF16));
        assert!(!muon.nesterov());
        assert!(muon.adaptive());
        assert_eq!(muon.preconditioning(), MuonPreconditioning::Aol);
        assert!(format!("{muon:?}").ends_with("dimension_numbers: true }"));
        assert!(matches!(
            ScaleByMuon::new().with_beta(1.0),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`beta` must be in `[0, 1)` but got `1`",
        ));
    }

    #[test]
    fn test_scale_by_muon_initialize() {
        let parameters = BTreeMap::from([("weights", Array::matrix(2, 2, vec![0.0f32; 4]).unwrap())]);
        let state = ScaleByMuon::new().initialize(&parameters).unwrap();
        assert_eq!(
            state.named_parameters().map(|(path, _)| path.to_string()).collect::<Vec<_>>(),
            vec!["$.count", "$.first_moments[\"weights\"]", "$.newton_schulz_coefficients"],
        );
        assert_eq!(state.newton_schulz_coefficients(), &Array::vector(vec![3.4445f32, -4.7750, 2.0315]).unwrap());
        assert_eq!(state_paths(&ScaleByMuon::new()).len(), 4);
    }

    #[test]
    fn test_scale_by_muon_update() {
        // The singular values `3` and `4` of `diag(3, 4)` are normalized by the Frobenius norm `5` to `0.6` and `0.8`
        // (the bias corrected momentum is a positive multiple of the gradient at the first step, which the
        // normalization removes), and then iterated five times with the standard coefficients.
        let muon = ScaleByMuon::new();
        let parameters = vec![Array::matrix(2, 2, vec![0.0f32; 4]).unwrap()];
        let arguments = UpdateArguments::new();
        let state = muon.initialize(&parameters).unwrap();
        let gradients = vec![Array::matrix(2, 2, vec![3.0f32, 0.0, 0.0, 4.0]).unwrap()];
        let (updates, state) = muon.update(gradients, state, &parameters, &arguments).unwrap();
        assert_close(&updates[0], &[iterate(0.6, &standard()), 0.0, 0.0, iterate(0.8, &standard())]);
        assert_close(&state.first_moments()[0], &[0.15, 0.0, 0.0, 0.2]);

        // At the second step with the gradient `diag(0, 1)`, the momentum is `diag(0.95 · 0.15, 0.95 · 0.2 + 0.05)`,
        // and the Nesterov estimate is `0.95 · μ / (1 - 0.95³) + 0.05 · g / (1 - 0.95²)`.
        let gradients = vec![Array::matrix(2, 2, vec![0.0f32, 0.0, 0.0, 1.0]).unwrap()];
        let (updates, state) = muon.update(gradients, state, &parameters, &arguments).unwrap();
        let first = 0.95 * (0.95 * 0.15) / (1.0 - 0.95f64.powi(3));
        let second = 0.95 * (0.95 * 0.2 + 0.05) / (1.0 - 0.95f64.powi(3)) + 0.05 / (1.0 - 0.95f64.powi(2));
        let norm = (first * first + second * second).sqrt();
        assert_close(&updates[0], &[iterate(first / norm, &standard()), 0.0, 0.0, iterate(second / norm, &standard())]);
        assert_eq!(state.count(), &Array::scalar(2i32).unwrap());

        // A rank-1 leaf has no default dimension numbers.
        let parameters = vec![Array::vector(vec![0.0f32; 2]).unwrap()];
        assert!(matches!(
            muon.update(parameters.clone(), muon.initialize(&parameters).unwrap(), &parameters, &arguments),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`ScaleByMuon` requires dimension numbers for the leaf at `$[0]`, which has rank 1",
        ));
    }

    #[test]
    fn test_scale_by_muon_update_shapes() {
        // The rank-1 matrix `M = [[1, 2, 2], [2, 4, 4]] = √45 · u vᵀ` (with unit `u = (1, 2) / √5` and
        // `v = (1, 2, 2) / 3`) has more columns than rows and the single singular value `√45`, which the normalization
        // maps to `1`, so the update is `p(1) · u vᵀ = p(1) · M / √45` with the iterated singular value `p(1)`.
        let muon = ScaleByMuon::new();
        let parameters = vec![Array::matrix(2, 3, vec![0.0f32; 6]).unwrap()];
        let gradients = vec![Array::matrix(2, 3, vec![1.0f32, 2.0, 2.0, 2.0, 4.0, 4.0]).unwrap()];
        let state = muon.initialize(&parameters).unwrap();
        let (updates, _) = muon.update(gradients.clone(), state, &parameters, &UpdateArguments::new()).unwrap();
        let scale = iterate(1.0, &standard()) / 45f64.sqrt();
        let expected = [1.0, 2.0, 2.0, 2.0, 4.0, 4.0].map(|element| element * scale);
        assert_close(&updates[0], &expected);

        // The transposed matrix has more rows than columns, and gives the transposed update.
        let parameters = vec![Array::matrix(3, 2, vec![0.0f32; 6]).unwrap()];
        let gradients = vec![Array::matrix(3, 2, vec![1.0f32, 2.0, 2.0, 4.0, 2.0, 4.0]).unwrap()];
        let state = muon.initialize(&parameters).unwrap();
        let (updates, _) = muon.update(gradients, state, &parameters, &UpdateArguments::new()).unwrap();
        assert_close(&updates[0], &[1.0, 2.0, 2.0, 4.0, 2.0, 4.0].map(|element| element * scale));

        // With the leading batch axis of a rank-3 leaf, every batch item is orthogonalized separately.
        let muon = ScaleByMuon::new().with_dimension_numbers(|_| Some(MuonDimensionNumbers::new(vec![1], vec![2])));
        let parameters =
            vec![Array::from_elements(ArrayType::new(DataType::F32, Shape::from([2, 2, 2])), &[0.0f32; 8]).unwrap()];
        let gradients = vec![
            Array::from_elements(
                ArrayType::new(DataType::F32, Shape::from([2, 2, 2])),
                &[3.0f32, 0.0, 0.0, 4.0, 1.0, 0.0, 0.0, 0.0],
            )
            .unwrap(),
        ];
        let state = muon.initialize(&parameters).unwrap();
        let (updates, _) = muon.update(gradients, state, &parameters, &UpdateArguments::new()).unwrap();
        let standard = standard();
        assert_close(
            &updates[0],
            &[iterate(0.6, &standard), 0.0, 0.0, iterate(0.8, &standard), iterate(1.0, &standard), 0.0, 0.0, 0.0],
        );
    }

    #[test]
    fn test_scale_by_muon_update_preconditioning() {
        // For `diag(3, 4)`, `X Xᵀ = diag(9, 16)`, whose absolute row sums rescale both rows to the singular value `1`
        // with the almost orthogonal layer scaling, and whose Frobenius norm `√337` rescales the singular values to
        // `3 / 337^¼` and `4 / 337^¼` with the Schatten scaling.
        let parameters = vec![Array::matrix(2, 2, vec![0.0f32; 4]).unwrap()];
        let gradients = vec![Array::matrix(2, 2, vec![3.0f32, 0.0, 0.0, 4.0]).unwrap()];
        let muon = ScaleByMuon::new().with_preconditioning(MuonPreconditioning::Aol);
        let state = muon.initialize(&parameters).unwrap();
        let (updates, _) = muon.update(gradients.clone(), state, &parameters, &UpdateArguments::new()).unwrap();
        let unit = iterate(1.0, &standard());
        assert_close(&updates[0], &[unit, 0.0, 0.0, unit]);
        let muon = ScaleByMuon::new().with_preconditioning(MuonPreconditioning::Schatten);
        let state = muon.initialize(&parameters).unwrap();
        let (updates, _) = muon.update(gradients, state, &parameters, &UpdateArguments::new()).unwrap();
        let root = 337f64.powf(0.25);
        assert_close(&updates[0], &[iterate(3.0 / root, &standard()), 0.0, 0.0, iterate(4.0 / root, &standard())]);
    }

    #[test]
    fn test_scale_by_muon_update_adaptive() {
        // Without Nesterov momentum, the first estimate is the gradient `diag(3, 4)` itself, so the adaptive scale is
        // `3 · p(0.6) + 4 · p(0.8)`.
        let muon = ScaleByMuon::new().with_nesterov(false).with_adaptive(true);
        let parameters = vec![Array::matrix(2, 2, vec![0.0f32; 4]).unwrap()];
        let gradients = vec![Array::matrix(2, 2, vec![3.0f32, 0.0, 0.0, 4.0]).unwrap()];
        let state = muon.initialize(&parameters).unwrap();
        let (updates, _) = muon.update(gradients, state, &parameters, &UpdateArguments::new()).unwrap();
        let (first, second) = (iterate(0.6, &standard()), iterate(0.8, &standard()));
        let scale = 3.0 * first + 4.0 * second;
        let actual = updates[0].elements::<f32>().unwrap();
        assert!((actual[0] as f64 - scale * first).abs() < 1e-3 && (actual[3] as f64 - scale * second).abs() < 1e-3);
    }

    #[test]
    fn test_scale_by_dimuon_new() {
        let dimuon = ScaleByDimuon::new();
        assert_eq!((dimuon.beta1(), dimuon.damping(), dimuon.rank_scaling_exponent()), (0.9, 1e-4, Some(0.175)));
        assert_eq!(ScaleByDimuon::default(), dimuon);
        assert_eq!(
            format!("{dimuon:?}"),
            "ScaleByDimuon { beta1: 0.9, damping: 0.0001, rank_scaling_exponent: Some(0.175) }",
        );
    }

    #[test]
    fn test_scale_by_dimuon_with_hyperparameters() {
        let dimuon = ScaleByDimuon::new()
            .with_beta1(0.8)
            .unwrap()
            .with_damping(0.01)
            .unwrap()
            .with_rank_scaling_exponent(None)
            .unwrap();
        assert_eq!((dimuon.beta1(), dimuon.damping(), dimuon.rank_scaling_exponent()), (0.8, 0.01, None));
        assert_eq!(ScaleByDimuon::new().with_beta1(0.0).unwrap().beta1(), 0.0);
        assert_eq!(ScaleByDimuon::new().with_damping(0.0).unwrap().damping(), 0.0);
        assert_eq!(
            ScaleByDimuon::new().with_rank_scaling_exponent(Some(0.5)).unwrap().rank_scaling_exponent(),
            Some(0.5),
        );
        assert!(matches!(
            ScaleByDimuon::new().with_beta1(1.0),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`beta1` must be in `[0, 1)` but got `1`",
        ));
        assert!(matches!(
            ScaleByDimuon::new().with_beta1(-0.1),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`beta1` must be in `[0, 1)` but got `-0.1`",
        ));
        assert!(matches!(
            ScaleByDimuon::new().with_damping(-1.0),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`damping` must be finite and non-negative but got `-1`",
        ));
        assert!(matches!(
            ScaleByDimuon::new().with_damping(f64::INFINITY),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`damping` must be finite and non-negative but got `inf`",
        ));
        assert!(matches!(
            ScaleByDimuon::new().with_rank_scaling_exponent(Some(f64::NAN)),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`rank_scaling_exponent` must be finite but got `NaN`",
        ));
    }

    #[test]
    fn test_scale_by_dimuon_initialize() {
        // The momentum mirrors the paths of the parameters and starts at zero.
        let parameters = BTreeMap::from([
            ("experts", reference_leaf(2, DataType::F32, &[1.0; 12])),
            ("weights", Array::matrix(3, 4, vec![1.0f32; 12]).unwrap()),
        ]);
        let state = ScaleByDimuon::new().initialize(&parameters).unwrap();
        assert_eq!(
            state.named_parameters().map(|(path, _)| path.to_string()).collect::<Vec<_>>(),
            vec!["$.first_moments[\"experts\"]", "$.first_moments[\"weights\"]"],
        );
        assert_eq!(state.first_moments()["weights"], Array::matrix(3, 4, vec![0.0f32; 12]).unwrap());
        assert_eq!(state.first_moments()["experts"], reference_leaf(2, DataType::F32, &[0.0; 12]));

        // Leaves of rank 0 and 1 are not LoRA factors.
        let message = |path: &str, rank: usize| {
            format!(
                "`ScaleByDimuon` requires leaves of rank at least 2 but the leaf at `{path}` has rank {rank}, so it \
                 should be routed to another optimizer with `Partition` or `Masked`",
            )
        };
        assert!(matches!(
            ScaleByDimuon::new().initialize(&BTreeMap::from([("bias", Array::scalar(1.0f32).unwrap())])),
            Err(OptimizerError::InvalidConfiguration { message: actual }) if actual == message("$[\"bias\"]", 0),
        ));
        assert!(matches!(
            ScaleByDimuon::new().initialize(&vec![Array::vector(vec![1.0f32, 2.0]).unwrap()]),
            Err(OptimizerError::InvalidConfiguration { message: actual }) if actual == message("$[0]", 1),
        ));
    }

    #[test]
    fn test_scale_by_dimuon_update() {
        // `dimuon_reference.py` (scratchpad `dimuon_reference/`, `float64` NumPy): `ScaleByDimuon` with its defaults
        // on the wide, tall, and rank-3 reference leaves for four steps.
        const DEFAULT_UPDATES: [[[f32; 12]; 3]; 4] = [
            [
                [
                    0.3601058,
                    0.4969881,
                    0.40012977,
                    0.115082696,
                    0.08605637,
                    -0.12837192,
                    -0.28242368,
                    -0.30364993,
                    -0.5222752,
                    -0.25507936,
                    0.13208498,
                    0.4571262,
                ],
                [
                    0.1780782,
                    0.35190174,
                    0.3602051,
                    0.35643646,
                    -0.07887687,
                    -0.47709656,
                    -0.53797436,
                    -0.2722527,
                    0.12151052,
                    0.1867533,
                    0.35376635,
                    0.3544096,
                ],
                [
                    0.49577266,
                    0.3593439,
                    0.05391001,
                    0.21867031,
                    -0.20836438,
                    -0.5374021,
                    -0.46222106,
                    -0.009695136,
                    0.44739053,
                    0.24003823,
                    0.39378014,
                    0.36232108,
                ],
            ],
            [
                [
                    0.6972223,
                    0.3973571,
                    0.060313042,
                    -0.12240859,
                    -0.14994577,
                    -0.29915258,
                    -0.19382837,
                    -0.12566799,
                    -0.46158904,
                    0.15742214,
                    0.31594548,
                    0.38010743,
                ],
                [
                    0.34225214,
                    0.25331345,
                    0.09264798,
                    0.12898482,
                    -0.32934898,
                    -0.6303134,
                    -0.50480354,
                    -0.03781032,
                    0.38921565,
                    0.34630382,
                    0.25081342,
                    0.085152425,
                ],
                [
                    0.55453026,
                    0.17276849,
                    -0.15534231,
                    -0.17507353,
                    -0.3832783,
                    -0.4677834,
                    -0.15965669,
                    0.18719836,
                    0.6811586,
                    0.3710085,
                    0.30794096,
                    0.029543974,
                ],
            ],
            [
                [
                    0.6850794,
                    0.33122927,
                    0.14664307,
                    -0.10922651,
                    -0.20331125,
                    -0.23969898,
                    -0.2382631,
                    -0.2196893,
                    -0.30860755,
                    0.12378321,
                    0.30992502,
                    0.5359113,
                ],
                [
                    0.32700658,
                    0.30072704,
                    0.15565583,
                    0.010014168,
                    -0.29016852,
                    -0.528683,
                    -0.51207453,
                    -0.06090742,
                    0.48684937,
                    0.33152378,
                    0.29966152,
                    0.14852756,
                ],
                [
                    0.52415425,
                    0.2123619,
                    -0.2563742,
                    -0.07608924,
                    -0.30334502,
                    -0.5239155,
                    -0.19248648,
                    0.23525941,
                    0.5462663,
                    0.4535482,
                    0.40001354,
                    -0.036559038,
                ],
            ],
            [
                [
                    -0.68236023,
                    -0.20853218,
                    0.03448069,
                    0.37451148,
                    0.30120564,
                    0.22316593,
                    0.17005618,
                    0.08241796,
                    0.11450772,
                    -0.21751828,
                    -0.36258298,
                    -0.539851,
                ],
                [
                    -0.35710856,
                    -0.19625464,
                    0.07000421,
                    0.09784787,
                    0.3033997,
                    0.4274334,
                    0.4074526,
                    -0.12649524,
                    -0.706705,
                    -0.36022353,
                    -0.19355567,
                    0.07859465,
                ],
                [
                    -0.4406513,
                    -0.043000266,
                    0.51206654,
                    0.19442196,
                    0.31889498,
                    0.400162,
                    -0.05364325,
                    -0.41467673,
                    -0.46095222,
                    -0.48981404,
                    -0.27191633,
                    0.23809804,
                ],
            ],
        ];
        let updates = reference_updates(&ScaleByDimuon::new(), DataType::F32);
        let precise = reference_updates(&ScaleByDimuon::new(), DataType::F64);
        assert_reference(&precise, &DEFAULT_UPDATES, 1e-6);
        assert_reference(&updates, &DEFAULT_UPDATES, REFERENCE_TOLERANCE);

        // Every matrix of every update has the Frobenius norm `√r · (r / 32)^0.175`, with `r = 3` for the 3 × 4 and
        // 4 × 3 leaves and `r = 2` for the 2 × 3 matrices of the rank-3 leaf.
        let norm = |rank: f64| rank.sqrt() * (rank / 32.0).powf(0.175);
        for step in &updates {
            let norms = [matrix_norms(&step[0], 1), matrix_norms(&step[1], 1), matrix_norms(&step[2], 2)].concat();
            let expected = [norm(3.0), norm(3.0), norm(2.0), norm(2.0)];
            assert!(norms.iter().zip(expected).all(|(norm, expected)| (norm - expected).abs() < 1e-5), "{norms:?}");
        }
    }

    #[test]
    fn test_scale_by_dimuon_update_first_step() {
        // On the first step, the previous momentum is zero, so the conditioners are the identity. The momentum
        // `0.1 · diag(3, 4)` of the gradient `diag(3, 4)` is normalized to the singular values `0.6 / 1.01` and
        // `0.8 / 1.01`, iterated five times, and rescaled to the Frobenius norm `√2 · (2 / 32)^0.175`.
        let parameters = vec![Array::matrix(2, 2, vec![0.0f32; 4]).unwrap()];
        let gradients = vec![Array::matrix(2, 2, vec![3.0f32, 0.0, 0.0, 4.0]).unwrap()];
        let (first, second) = (polar_express(0.6 / 1.01), polar_express(0.8 / 1.01));
        let scale = 2f64.sqrt() * (2.0f64 / 32.0).powf(0.175) / (first * first + second * second).sqrt();
        let expected = vec![vec![(first * scale) as f32, 0.0, 0.0, (second * scale) as f32]];
        for dimuon in [ScaleByDimuon::new(), ScaleByDimuon::new().with_damping(0.5).unwrap()] {
            let state = dimuon.initialize(&parameters).unwrap();
            let (updates, state) =
                dimuon.update(gradients.clone(), state, &parameters, &UpdateArguments::new()).unwrap();
            assert_parity_with_tolerance(&[updates[0].elements::<f32>().unwrap()], &expected, 1e-5);
            assert_eq!(state.first_moments()[0], Array::matrix(2, 2, vec![0.3f32, 0.0, 0.0, 0.4]).unwrap());
        }
    }

    #[test]
    fn test_scale_by_dimuon_update_without_rank_scaling() {
        // `dimuon_reference.py`: `ScaleByDimuon` with the damping `0.01` and without the rank factor, whose updates
        // have the Frobenius norm `√r`.
        const UNSCALED_UPDATES: [[[f32; 12]; 3]; 4] = [
            [
                [
                    0.544923,
                    0.75205743,
                    0.6054885,
                    0.17414662,
                    0.1302231,
                    -0.19425628,
                    -0.42737207,
                    -0.45949227,
                    -0.7903227,
                    -0.38599378,
                    0.199875,
                    0.69173723,
                ],
                [
                    0.26947334,
                    0.5325084,
                    0.5450733,
                    0.5393705,
                    -0.11935887,
                    -0.72195697,
                    -0.8140791,
                    -0.41198102,
                    0.1838734,
                    0.28260076,
                    0.53532994,
                    0.53630334,
                ],
                [
                    0.80538505,
                    0.58375585,
                    0.08757707,
                    0.355231,
                    -0.33848894,
                    -0.8730123,
                    -0.7508803,
                    -0.015749795,
                    0.72678804,
                    0.38994324,
                    0.63969773,
                    0.58859235,
                ],
            ],
            [
                [
                    1.0490264,
                    0.60409033,
                    0.090427,
                    -0.19187617,
                    -0.22202244,
                    -0.44933084,
                    -0.29447564,
                    -0.19088295,
                    -0.70076597,
                    0.22872476,
                    0.48111907,
                    0.5834528,
                ],
                [
                    0.52080625,
                    0.38834575,
                    0.14471175,
                    0.18717988,
                    -0.5009716,
                    -0.9508,
                    -0.7604435,
                    -0.05851876,
                    0.5848242,
                    0.52688324,
                    0.38453412,
                    0.13336712,
                ],
                [
                    0.9004365,
                    0.28143093,
                    -0.2534205,
                    -0.28229034,
                    -0.6226526,
                    -0.7605292,
                    -0.2285381,
                    0.3027622,
                    1.0420955,
                    0.6683366,
                    0.5637788,
                    0.07497959,
                ],
            ],
            [
                [
                    1.0411503,
                    0.49386117,
                    0.20161772,
                    -0.20093538,
                    -0.28828305,
                    -0.34690508,
                    -0.34560126,
                    -0.31733257,
                    -0.5088567,
                    0.16427258,
                    0.4607813,
                    0.8180782,
                ],
                [
                    0.49718568,
                    0.45755723,
                    0.23682016,
                    0.01375647,
                    -0.4409285,
                    -0.80026907,
                    -0.7715924,
                    -0.09207221,
                    0.73179144,
                    0.50401086,
                    0.45592028,
                    0.22602025,
                ],
                [
                    0.85182685,
                    0.34323895,
                    -0.41904357,
                    -0.12151193,
                    -0.49280244,
                    -0.85050637,
                    -0.31937227,
                    0.36772522,
                    0.8770074,
                    0.7509683,
                    0.6517937,
                    -0.069635555,
                ],
            ],
            [
                [
                    -1.0277638,
                    -0.31890503,
                    0.05269936,
                    0.56934226,
                    0.45178214,
                    0.33947566,
                    0.25868586,
                    0.12416248,
                    0.17611472,
                    -0.32908043,
                    -0.5516887,
                    -0.81836236,
                ],
                [
                    -0.54423565,
                    -0.30255798,
                    0.1002523,
                    0.15534735,
                    0.4670328,
                    0.65121084,
                    0.60806084,
                    -0.19342303,
                    -1.0609614,
                    -0.5488477,
                    -0.29838985,
                    0.11326031,
                ],
                [
                    -0.71909773,
                    -0.07650588,
                    0.8189509,
                    0.32884037,
                    0.5270674,
                    0.6484047,
                    -0.089581475,
                    -0.6654735,
                    -0.73847264,
                    -0.8073465,
                    -0.44824177,
                    0.38865054,
                ],
            ],
        ];
        let dimuon = ScaleByDimuon::new().with_damping(0.01).unwrap().with_rank_scaling_exponent(None).unwrap();
        let updates = reference_updates(&dimuon, DataType::F32);
        let precise = reference_updates(&dimuon, DataType::F64);
        assert_reference(&precise, &UNSCALED_UPDATES, 1e-6);
        assert_reference(&updates, &UNSCALED_UPDATES, REFERENCE_TOLERANCE);
        for step in &updates {
            let norms = [matrix_norms(&step[0], 1), matrix_norms(&step[1], 1), matrix_norms(&step[2], 2)].concat();
            let expected = [3f64.sqrt(), 3f64.sqrt(), 2f64.sqrt(), 2f64.sqrt()];
            assert!(norms.iter().zip(expected).all(|(norm, expected)| (norm - expected).abs() < 1e-5), "{norms:?}");
        }
    }

    #[test]
    fn test_scale_by_dimuon_update_zero() {
        // A zero gradient with a zero momentum gives a zero update rather than NaNs, matrix by matrix, so the nonzero
        // matrix of the rank-3 leaf is updated as on its own.
        let dimuon = ScaleByDimuon::new();
        let parameters =
            vec![reference_leaf(2, DataType::F32, &[0.0; 12]), Array::matrix(3, 4, vec![0.0f32; 12]).unwrap()];
        let gradients = vec![
            reference_leaf(2, DataType::F32, &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0, 0.0]),
            Array::matrix(3, 4, vec![0.0f32; 12]).unwrap(),
        ];
        let state = dimuon.initialize(&parameters).unwrap();
        let (updates, state) = dimuon.update(gradients, state, &parameters, &UpdateArguments::new()).unwrap();
        let (first, second) = (polar_express(0.6 / 1.01), polar_express(0.8 / 1.01));
        let scale = 2f64.sqrt() * (2.0f64 / 32.0).powf(0.175) / (first * first + second * second).sqrt();
        let (first, second) = ((first * scale) as f32, (second * scale) as f32);
        assert_parity_with_tolerance(
            &[updates[0].elements::<f32>().unwrap()],
            &[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, first, 0.0, 0.0, 0.0, second, 0.0]],
            1e-5,
        );
        assert_eq!(updates[1], Array::matrix(3, 4, vec![0.0f32; 12]).unwrap());

        // With zero gradients on the next step, the matrices without momentum stay zero, and the nonzero matrix of the
        // rank-3 leaf keeps an update of the Frobenius norm `√2 · (2 / 32)^0.175`, because its momentum is not zero.
        let gradients =
            vec![reference_leaf(2, DataType::F32, &[0.0; 12]), Array::matrix(3, 4, vec![0.0f32; 12]).unwrap()];
        let (updates, _) = dimuon.update(gradients, state, &parameters, &UpdateArguments::new()).unwrap();
        let norms = matrix_norms(&updates[0].elements::<f32>().unwrap(), 2);
        assert_eq!(norms[0], 0.0);
        assert!((norms[1] - 2f64.sqrt() * (2.0f64 / 32.0).powf(0.175)).abs() < 1e-5, "{norms:?}");
        assert_eq!(updates[1], Array::matrix(3, 4, vec![0.0f32; 12]).unwrap());
    }

    #[test]
    fn test_scale_by_dimuon_update_ranks() {
        // Leaves of rank 0 and 1 are rejected by updates too, even with a state that was not created by `initialize`.
        let dimuon = ScaleByDimuon::new();
        for (leaf, rank) in [(Array::scalar(1.0f32).unwrap(), 0), (Array::vector(vec![1.0f32, 2.0]).unwrap(), 1)] {
            let parameters = vec![leaf.clone()];
            let state = ScaleByDimuonState { first_moments: parameters.clone(), leaf: PhantomData };
            assert!(matches!(
                dimuon.update(parameters.clone(), state, &parameters, &UpdateArguments::new()),
                Err(OptimizerError::InvalidConfiguration { message }) if message == format!(
                    "`ScaleByDimuon` requires leaves of rank at least 2 but the leaf at `$[0]` has rank {rank}, so it \
                     should be routed to another optimizer with `Partition` or `Masked`",
                ),
            ));
        }
    }

    #[test]
    fn test_scale_by_dimuon_update_data_types() {
        // The momentum of `f64` parameters is created in `f64` and stays `f64` after an update with `f32` gradients,
        // whose updates are `f32`, so that compiled training steps see the same state types at every step.
        let dimuon = ScaleByDimuon::new();
        let parameters = vec![Array::matrix(2, 2, vec![1.0f64, 2.0, 3.0, 4.0]).unwrap()];
        let gradients = vec![Array::matrix(2, 2, vec![0.5f32, -0.25, 0.75, 1.0]).unwrap()];
        let state = dimuon.initialize(&parameters).unwrap();
        assert_eq!(state.first_moments()[0].r#type().data_type(), DataType::F64);
        let (updates, state) = dimuon.update(gradients.clone(), state, &parameters, &UpdateArguments::new()).unwrap();
        assert_eq!(updates[0].r#type().data_type(), DataType::F32);
        assert_eq!(state.first_moments()[0].r#type().data_type(), DataType::F64);
        let (_, state) = dimuon.update(gradients, state, &parameters, &UpdateArguments::new()).unwrap();
        assert_eq!(state.first_moments()[0].r#type().data_type(), DataType::F64);
    }
}
