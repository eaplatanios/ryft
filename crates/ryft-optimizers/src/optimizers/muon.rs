//! Optimizers that orthogonalize the momentum of matrix parameters.
//!
//!   - **Muon:** [`Muon`] orthogonalizes the momentum of the matrix leaves with Newton–Schulz iterations (refer to
//!     [`ScaleByMuon`]) and updates all other leaves with AdamW, in the style of
//!     [`optax.contrib.muon`](https://optax.readthedocs.io/en/latest/api/contrib.html#optax.contrib.muon).
//!   - **Dimuon:** [`Dimuon`] orthogonalizes the row and column conditioned momentum of LoRA factors (refer to
//!     [`ScaleByDimuon`]) and adds global gradient clipping and a learning rate. It is an unofficial reconstruction of
//!     the optimizer of Thinking Machines, whose formulation is unpublished.

use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::marker::PhantomData;

use ryft_core::{DataType, Mul, Parameter, ParameterPath, Parameterized};

use crate::macros::parameterized_state;
use crate::transformations::muon::static_shape;
use crate::{
    AdamW, AdamWState, AddDecayedWeights, ClipByGlobalNorm, LeafPath, MuonDimensionNumbers, Optimizer, OptimizerError,
    OptimizerValue, ScaleByDimuon, ScaleByDimuonState, ScaleByMuon, ScaleByMuonState, ScaleBySchedule,
    ScaleByScheduleState, Schedule, SelectedLeaves, UpdateArguments, convert_to, statistic_data_type,
    validate_non_negative, validate_structure,
};

// TODO(eaplatanios): Review this module.

parameterized_state! {
    /// State of [`Muon`] for the parameter structure `S`, whose fields hold one element when the optimizer has leaves
    /// of the corresponding kind and none otherwise. The states of the parts of the optimizer hold their statistics for
    /// the [`SelectedLeaves`] of their leaves, so their leaf paths name the parameters by their paths (e.g.,
    /// `$.muon[0].first_moments[$.weights]`).
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct MuonState<P, S> with family MuonStateFamily and marker leaf {
        /// State of the [`ScaleByMuon`] transformation of the matrix leaves.
        muon: Vec<ScaleByMuonState<P, SelectedLeaves<P>>>,

        /// State of the [`ScaleBySchedule`] transformation that applies the learning rate to the matrix leaves.
        learning_rate: Vec<ScaleByScheduleState<P>>,

        /// State of the [`AdamW`] optimizer of the other leaves.
        adam: Vec<AdamWState<P, SelectedLeaves<P>>>,
    }
}

impl<P: Parameter, S> MuonState<P, S> {
    /// Returns the state of the [`ScaleByMuon`] transformation of the matrix leaves, if there are any.
    #[inline]
    pub fn muon(&self) -> Option<&ScaleByMuonState<P, SelectedLeaves<P>>> {
        self.muon.first()
    }

    /// Returns the state of the [`ScaleBySchedule`] transformation that applies the learning rate to the matrix
    /// leaves, if there are any.
    #[inline]
    pub fn learning_rate(&self) -> Option<&ScaleByScheduleState<P>> {
        self.learning_rate.first()
    }

    /// Returns the state of the [`AdamW`] optimizer of the other leaves, if there are any.
    #[inline]
    pub fn adam(&self) -> Option<&AdamWState<P, SelectedLeaves<P>>> {
        self.adam.first()
    }
}

/// Muon optimizer of [Jordan et al. (2024)](https://kellerjordan.github.io/posts/muon/), which updates every matrix
/// leaf with its orthogonalized momentum and all other leaves with AdamW, and which is the analogue of
/// [`optax.contrib.muon`](https://optax.readthedocs.io/en/latest/api/contrib.html#optax.contrib.muon). For a matrix
/// leaf, its update is
///
/// ```text
/// μₜ = β · μₜ₋₁ + (1 − β) · gₜ
/// μ̂ₜ = β · μₜ / (1 − βᵗ⁺¹) + (1 − β) · gₜ / (1 − βᵗ)
/// X₀ = μ̂ₜ / (‖μ̂ₜ‖ + ε)
/// Aₖ = Xₖ₋₁ · Xₖ₋₁ᵀ
/// Xₖ = a · Xₖ₋₁ + (b · Aₖ + c · Aₖ²) · Xₖ₋₁   for k = 1, …, n
/// ```
///
/// followed by
///
/// ```text
/// uₜ = −ηₜ · (s · Xₙ + λ · θₜ₋₁)
/// ```
///
/// Here `t ≥ 1` counts the updates including the current one, `gₜ` is the gradient at the parameters `θₜ₋₁`, `uₜ` is
/// the update that is added to them (i.e., `θₜ = θₜ₋₁ + uₜ`), and `ηₜ` is the learning rate [`Schedule`] evaluated at
/// the `t − 1` updates applied before the current one. The momentum `μₜ` starts at zero and decays with the rate `β`,
/// and `μ̂ₜ` is its Nesterov estimate. Every matrix of `μ̂ₜ`, whose rows and columns are given by
/// [`MuonDimensionNumbers`], is transposed first when it has more rows than columns (and transposed back at the end),
/// `‖·‖` is its Frobenius norm, `ε` keeps the division finite, and the `n` Newton–Schulz iterations with the
/// coefficients `(a, b, c)` drive its singular values towards one, so `Xₙ` approximates the orthogonal factor of `μ̂ₜ`.
/// The products in `Aₖ` and `Xₖ` are matrix products, and all other operations are element-wise. The shape scaling is
/// `s = √max(1, fan out / fan in)` for the extents of the columns (i.e., the fan out) and of the rows (i.e., the fan
/// in) of the matrix, and `λ` is the rate of the decoupled weight decay. The matrix leaves are the rank-2 leaves or,
/// with a dimension numbers function of [`ScaleByMuon`], the leaves for which that function returns dimension numbers.
/// All other leaves are updated with [`AdamW`], whose default is Nesterov momentum with `β₁ = 0.9`, `β₂ = 0.999`, the
/// `ε` of Muon, `ε̄ = 0`, no weight decay, and the learning rate of Muon. The defaults of the matrix leaves are those
/// of [`ScaleByMuon`] (e.g., `β = 0.95` and `n = 5`) and `λ = 0`. [`Self::with_consistent_root_mean_square`] replaces
/// the shape scaling with `s = ϱ · √max(fan in, fan out)` for the target root mean square `ϱ` of Liu et al. (2025), and
/// [`Self::with_weight_decay`] and [`Self::with_weight_decay_mask`] set `λ` and the matrix leaves that decay.
///
/// [`Self::with_epsilon`], [`Self::with_nesterov`], and [`Self::with_first_moment_data_type`] set the hyperparameters
/// that Optax shares between Muon and AdamW for both, and [`Self::with_adam`] replaces the AdamW optimizer (e.g., to
/// change its decay rates, its weight decay, or its learning rate). Optax applies the two parts with `optax.partition`,
/// which this optimizer replaces with its own selection, because the partition depends on the ranks of the leaves
/// rather than only on their paths. It needs no [`UpdateArguments`] and passes none to its parts.
///
/// The matrix leaves are updated with the chain of [`ScaleByMuon`], the shape scaling, [`AddDecayedWeights`], and
/// [`ScaleBySchedule::learning_rate`].
///
/// # References
///
///   - Jordan et al. (2024). [Muon: An optimizer for hidden layers in neural
///     networks](https://kellerjordan.github.io/posts/muon/).
///   - Bernstein and Newhouse (2024). [Old Optimizer, New Norm: An Anthology](https://arxiv.org/abs/2409.20325).
///   - Liu et al. (2025). [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982).
#[derive(Clone)]
pub struct Muon<Sc: Schedule + Clone = f64> {
    /// Transformation that orthogonalizes the momentum of the matrix leaves.
    muon: ScaleByMuon,

    /// Root mean square to which the shape scaling normalizes the updates, or [`None`] for the width transfer scaling.
    consistent_root_mean_square: Option<f64>,

    /// Weight decay of the matrix leaves.
    weight_decay: AddDecayedWeights,

    /// Learning rate schedule of the matrix leaves.
    learning_rate: ScaleBySchedule<Sc>,

    /// Optimizer of the leaves that are not matrices.
    adam: AdamW<Sc>,
}

impl<Sc: Schedule + Clone> Muon<Sc> {
    /// Creates a new [`Muon`] with the learning rate schedule `learning_rate` (e.g., an `f64` for a constant learning
    /// rate) and default values for all other hyperparameters.
    pub fn new(learning_rate: Sc) -> Self {
        let adam = AdamW::new(learning_rate.clone())
            .with_epsilon(1e-8)
            .unwrap()
            .with_weight_decay(0.0)
            .unwrap()
            .with_nesterov(true);
        Self {
            muon: ScaleByMuon::new(),
            consistent_root_mean_square: None,
            weight_decay: AddDecayedWeights::new(0.0).unwrap(),
            learning_rate: ScaleBySchedule::learning_rate(learning_rate),
            adam,
        }
    }

    /// Returns this optimizer with `muon` as the transformation of its matrix leaves (e.g., to change its coefficients,
    /// its momentum, or its dimension numbers), whose dimension numbers function also selects the matrix leaves.
    #[inline]
    pub fn with_muon(mut self, muon: ScaleByMuon) -> Self {
        self.muon = muon;
        self
    }

    /// Returns this optimizer with `adam` as the optimizer of the leaves that are not matrices.
    #[inline]
    pub fn with_adam(mut self, adam: AdamW<Sc>) -> Self {
        self.adam = adam;
        self
    }

    /// Returns this optimizer with `epsilon` as the `ε` of both its Muon and its AdamW parts.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `epsilon` is negative or not finite.
    pub fn with_epsilon(mut self, epsilon: f64) -> Result<Self, OptimizerError> {
        self.muon = self.muon.with_epsilon(epsilon)?;
        self.adam = self.adam.with_epsilon(epsilon)?;
        Ok(self)
    }

    /// Returns this optimizer with Nesterov momentum in both its Muon and its AdamW parts when `nesterov` is `true`.
    #[inline]
    pub fn with_nesterov(mut self, nesterov: bool) -> Self {
        self.muon = self.muon.with_nesterov(nesterov);
        self.adam = self.adam.with_nesterov(nesterov);
        self
    }

    /// Returns this optimizer with the first moments of both its Muon and its AdamW parts stored in `data_type`.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `data_type` is not a floating-point data type.
    pub fn with_first_moment_data_type(mut self, data_type: DataType) -> Result<Self, OptimizerError> {
        self.muon = self.muon.with_first_moment_data_type(data_type)?;
        self.adam = self.adam.with_first_moment_data_type(data_type)?;
        Ok(self)
    }

    /// Returns this optimizer with `consistent_root_mean_square` as the root mean square to which its shape scaling
    /// normalizes the updates of the matrix leaves, or with the width transfer scaling when it is [`None`].
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when the root mean square is negative or not finite.
    pub fn with_consistent_root_mean_square(
        mut self,
        consistent_root_mean_square: Option<f64>,
    ) -> Result<Self, OptimizerError> {
        if let Some(root_mean_square) = consistent_root_mean_square {
            validate_non_negative("consistent_root_mean_square", root_mean_square)?;
        }
        self.consistent_root_mean_square = consistent_root_mean_square;
        Ok(self)
    }

    /// Returns this optimizer with `weight_decay` as the rate of the weight decay of its matrix leaves, keeping its
    /// mask.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `weight_decay` is not finite.
    pub fn with_weight_decay(mut self, weight_decay: f64) -> Result<Self, OptimizerError> {
        self.weight_decay = self.weight_decay.with_rate(weight_decay)?;
        Ok(self)
    }

    /// Returns this optimizer with `mask` as the predicate that selects the matrix leaves that decay by their
    /// [`ParameterPath`]s in the parameters (refer to [`AddDecayedWeights::with_mask`]).
    pub fn with_weight_decay_mask<Mask>(mut self, mask: Mask) -> Self
    where
        Mask: 'static + Fn(&ParameterPath) -> bool + Send + Sync,
    {
        self.weight_decay = self.weight_decay.with_mask(mask);
        self
    }

    /// Returns the [`ScaleByMuon`] transformation of the matrix leaves of this optimizer.
    #[inline]
    pub fn muon(&self) -> &ScaleByMuon {
        &self.muon
    }

    /// Returns the [`AdamW`] optimizer of the leaves of this optimizer that are not matrices.
    #[inline]
    pub fn adam(&self) -> &AdamW<Sc> {
        &self.adam
    }

    /// Returns the root mean square to which the shape scaling normalizes the updates of the matrix leaves, if any.
    #[inline]
    pub fn consistent_root_mean_square(&self) -> Option<f64> {
        self.consistent_root_mean_square
    }

    /// Returns the [`AddDecayedWeights`] transformation of the matrix leaves of this optimizer.
    #[inline]
    pub fn weight_decay(&self) -> &AddDecayedWeights {
        &self.weight_decay
    }

    /// Returns the learning rate [`Schedule`] of the matrix leaves of this optimizer.
    #[inline]
    pub fn learning_rate(&self) -> &Sc {
        self.learning_rate.schedule()
    }

    /// Returns the partition of the leaves of `structure` into the [`SelectedLeaves`] of the matrix leaves, with the
    /// dimension numbers of every matrix leaf, and the [`SelectedLeaves`] of all other leaves.
    #[allow(clippy::type_complexity)]
    fn partition<P: OptimizerValue, S: Parameterized<P>>(
        &self,
        structure: &S,
    ) -> Result<(SelectedLeaves<P>, HashMap<ParameterPath, MuonDimensionNumbers>, SelectedLeaves<P>), OptimizerError>
    {
        let mut matrices = SelectedLeaves::new();
        let mut dimension_numbers = HashMap::new();
        let mut others = SelectedLeaves::new();
        for (path, leaf) in structure.named_parameters() {
            match self.muon.matrix_dimension_numbers(&path, leaf.r#type().rank()) {
                Some(numbers) => {
                    dimension_numbers.insert(ParameterPath::root().key(LeafPath::new(path.clone())), numbers);
                    matrices.insert(LeafPath::new(path), leaf.clone());
                }
                None => {
                    others.insert(LeafPath::new(path), leaf.clone());
                }
            }
        }
        Ok((matrices, dimension_numbers, others))
    }

    /// Returns the [`ScaleByMuon`] transformation of the [`SelectedLeaves`] of the matrix leaves, whose dimension
    /// numbers are `dimension_numbers` by the paths of those leaves.
    fn selected_muon(&self, dimension_numbers: HashMap<ParameterPath, MuonDimensionNumbers>) -> ScaleByMuon {
        self.muon.clone().with_dimension_numbers(move |path| dimension_numbers.get(path).cloned())
    }

    /// Returns the [`AddDecayedWeights`] transformation of the [`SelectedLeaves`] of the matrix leaves, which applies
    /// the mask of this optimizer to the paths of the parameters.
    fn selected_weight_decay<P: Parameter>(
        &self,
        matrices: &SelectedLeaves<P>,
    ) -> Result<AddDecayedWeights, OptimizerError> {
        let decaying = matrices
            .keys()
            .filter(|path| self.weight_decay.decays(path.path()))
            .map(|path| ParameterPath::root().key(path.clone()))
            .collect::<HashSet<_>>();
        Ok(AddDecayedWeights::new(self.weight_decay.rate())?.with_mask(move |path| decaying.contains(path)))
    }

    /// Returns `updates` scaled by the shapes of their leaves (i.e., `scale_by_shape` of Optax).
    fn scale_by_shape<P: OptimizerValue>(
        &self,
        updates: SelectedLeaves<P>,
        dimension_numbers: &HashMap<ParameterPath, MuonDimensionNumbers>,
    ) -> Result<SelectedLeaves<P>, OptimizerError> {
        updates
            .into_iter()
            .map(|(path, update)| {
                let shape = static_shape("ScaleByMuon", &update)?;
                let (_, reduction_axes, output_axes) =
                    dimension_numbers[&ParameterPath::root().key(path.clone())].resolve(&shape)?;
                let fan_in = reduction_axes.iter().map(|axis| shape[*axis]).product::<usize>() as f64;
                let fan_out = output_axes.iter().map(|axis| shape[*axis]).product::<usize>() as f64;
                let scale = match self.consistent_root_mean_square {
                    Some(root_mean_square) => fan_in.max(fan_out).sqrt() * root_mean_square,
                    None => (fan_out / fan_in).max(1.0).sqrt(),
                };
                let data_type = update.r#type().data_type();
                let statistic_type = statistic_data_type(data_type);
                let update = convert_to(&update, statistic_type)?;
                Ok((path, convert_to(&Mul::mul(&update.fill_like(scale)?, &update)?, data_type)?))
            })
            .collect()
    }
}

impl<Sc: Schedule + Clone + Debug> Debug for Muon<Sc> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Muon")
            .field("muon", &self.muon)
            .field("consistent_root_mean_square", &self.consistent_root_mean_square)
            .field("weight_decay", &self.weight_decay)
            .field("learning_rate", &self.learning_rate)
            .field("adam", &self.adam)
            .finish()
    }
}

impl<P: OptimizerValue, Sc: Schedule + Clone> Optimizer<P> for Muon<Sc> {
    type State<S: Parameterized<P>> = MuonState<P, S>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        let (matrices, dimension_numbers, others) = self.partition(parameters)?;
        let mut state = MuonState { muon: Vec::new(), learning_rate: Vec::new(), adam: Vec::new(), leaf: PhantomData };
        if !matrices.is_empty() {
            state.muon.push(self.selected_muon(dimension_numbers).initialize(&matrices)?);
            state.learning_rate.push(self.learning_rate.initialize(&matrices)?);
        }
        if !others.is_empty() {
            state.adam.push(self.adam.initialize(&others)?);
        }
        Ok(state)
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        // This follows `optax.contrib.muon`, whose partition is applied by selecting the leaves of each part.
        let _ = arguments;
        let paths = gradients.parameter_paths().collect::<Vec<_>>();
        validate_structure("parameters", &paths, parameters)?;
        let (mut matrix_updates, dimension_numbers, mut other_updates) = self.partition(&gradients)?;
        let (matrix_parameters, _, other_parameters) = self.partition(parameters)?;
        let MuonState { muon, learning_rate, adam, .. } = state;
        let mut next_state =
            MuonState { muon: Vec::new(), learning_rate: Vec::new(), adam: Vec::new(), leaf: PhantomData };
        let empty = UpdateArguments::new();
        if let (Some(muon), Some(learning_rate)) = (muon.into_iter().next(), learning_rate.into_iter().next()) {
            let weight_decay = self.selected_weight_decay(&matrix_parameters)?;
            let (updates, muon) = self.selected_muon(dimension_numbers.clone()).update(
                matrix_updates,
                muon,
                &matrix_parameters,
                &empty,
            )?;
            let updates = self.scale_by_shape(updates, &dimension_numbers)?;
            let (updates, ()) = weight_decay.update(updates, (), &matrix_parameters, &empty)?;
            let (updates, learning_rate) =
                self.learning_rate.update(updates, learning_rate, &matrix_parameters, &empty)?;
            matrix_updates = updates;
            next_state.muon.push(muon);
            next_state.learning_rate.push(learning_rate);
        } else if !matrix_updates.is_empty() {
            return Err(OptimizerError::StructureMismatch {
                message: "expected the state of `Muon` to have a state for its matrix leaves".to_string(),
            });
        }
        if let Some(adam) = adam.into_iter().next() {
            let (updates, adam) = self.adam.update(other_updates, adam, &other_parameters, &empty)?;
            other_updates = updates;
            next_state.adam.push(adam);
        } else if !other_updates.is_empty() {
            return Err(OptimizerError::StructureMismatch {
                message: "expected the state of `Muon` to have a state for its leaves that are not matrices"
                    .to_string(),
            });
        }

        let structure = gradients.parameter_structure();
        let updates = gradients
            .into_named_parameters()
            .map(|(path, _)| {
                let path = LeafPath::new(path);
                matrix_updates.remove(&path).or_else(|| other_updates.remove(&path)).ok_or_else(|| {
                    OptimizerError::StructureMismatch {
                        message: format!("expected an update for the leaf at `{path}`"),
                    }
                })
            })
            .collect::<Result<Vec<_>, OptimizerError>>()?;
        Ok((S::from_parameters(structure, updates)?, next_state))
    }
}

parameterized_state! {
    /// State of [`Dimuon`] for the parameter structure `S`, which holds the state of its [`ScaleByDimuon`]
    /// transformation and the step count of its learning rate schedule. The global gradient clipping is stateless.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct DimuonState<P, S> with family DimuonStateFamily {
        /// State of the [`ScaleByDimuon`] transformation of the optimizer.
        dimuon: ScaleByDimuonState<P, S>,

        /// State of the [`ScaleBySchedule`] transformation that applies the learning rate.
        learning_rate: ScaleByScheduleState<P>,
    }
}

impl<P: Parameter, S> DimuonState<P, S> {
    /// Returns the state of the [`ScaleByDimuon`] transformation of the optimizer.
    #[inline]
    pub fn dimuon(&self) -> &ScaleByDimuonState<P, S> {
        &self.dimuon
    }

    /// Returns the state of the [`ScaleBySchedule`] transformation that applies the learning rate.
    #[inline]
    pub fn learning_rate(&self) -> &ScaleByScheduleState<P> {
        &self.learning_rate
    }
}

/// Dimuon optimizer, which Thinking Machines built for LoRA fine tuning and serves through its Tinker API, and which
/// updates every LoRA factor with its orthogonalized momentum after conditioning the rows and the columns of that
/// momentum by their energies. Its official formulation is unpublished, so this is the unofficial reconstruction of
/// [anpaure/dimuon](https://github.com/anpaure/dimuon), which may differ from the optimizer that the Tinker API serves.
/// For every matrix `W` of shape `m × n` of a leaf, its update is
///
/// ```text
/// cₜ  = min(1, κ / ‖gₜ‖)
/// eʳᵢ = Σⱼ (Mₜ₋₁)ᵢⱼ²
/// eᶜⱼ = Σᵢ (Mₜ₋₁)ᵢⱼ²
/// L   = diag((eʳ / max eʳ + δ)^(−1/4))
/// R   = diag((eᶜ / max eᶜ + δ)^(−1/4))
/// Mₜ  = β₁ · Mₜ₋₁ + (1 − β₁) · cₜ · gₜ
/// Pₜ  = polar(L · Mₜ · R)
/// Dₜ  = L · Pₜ · R
/// uₜ  = −ηₜ · √r · (r / 32)^p · Dₜ / ‖Dₜ‖
/// ```
///
/// Here `t ≥ 1` counts the updates including the current one, `gₜ` is the gradient at the parameters `θₜ₋₁`, `uₜ` is
/// the update that is added to them (i.e., `θₜ = θₜ₋₁ + uₜ`), and `ηₜ` is the learning rate [`Schedule`] evaluated at
/// the `t − 1` updates applied before the current one, which is typically 10 to 100 times the learning rate of AdamW.
/// `cₜ` is the global gradient clipping scale for the maximum norm `κ` and the global norm `‖gₜ‖` of the gradients of
/// all leaves, or `1` without clipping. The row energies `eʳ` and the column energies `eᶜ` are those of the previous
/// momentum `Mₜ₋₁`, which starts at zero, so the diagonal conditioners `L` and `R` are the identity on the first step
/// (and whenever `max eʳ` or `max eᶜ` is zero). `δ` is the relative damping of the energies, `β₁` is the decay of the
/// momentum, `r = min(m, n)` is the rank of the matrix, `p` is the exponent of its rank scaling relative to the
/// reference rank `32`, and `‖Dₜ‖` is the Frobenius norm. The update is zero when `L · Mₜ · R` is zero. `polar` is five
/// Polar Express iterations, which drive the singular values of a matrix towards one (refer to [`ScaleByDimuon`]). The
/// products of `L`, `R`, `Mₜ`, and `Pₜ` are matrix products, and all other operations are element-wise. The defaults
/// are `β₁ = 0.9`, `δ = 1e-4`, `p = 0.175`, and no clipping (i.e., `cₜ = 1`), which are the defaults of the Tinker API
/// (whose `grad_clip_norm = 0` disables clipping). [`Self::with_beta1`], [`Self::with_damping`], and
/// [`Self::with_rank_scaling_exponent`] set `β₁`, `δ`, and `p` (refer to [`ScaleByDimuon`]), and
/// [`Self::with_gradient_clipping_norm`] sets `κ`. There is no second moment, no bias correction, and no weight decay.
///
/// It is the chain of [`ClipByGlobalNorm`] (only with a clipping norm), [`ScaleByDimuon`], and
/// [`ScaleBySchedule::learning_rate`]. [`ClipByGlobalNorm`] rescales the gradients by `cₜ` before they enter the
/// momentum, which is where the reconstruction applies its clipping scale, and its global norm covers the leaves that
/// this optimizer receives (e.g., only the LoRA factors when other leaves are routed elsewhere with
/// [`Partition`](crate::Partition)). Leaves of rank at least 3 are batches of matrices over their leading dimensions,
/// and leaves of rank 0 or 1 are rejected. Statistics are computed in `f32` (or `f64` for `f64` gradients) rather than
/// in the `bf16` momentum and `fp16` polar iterations of the reconstruction.
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
#[derive(Clone, Debug)]
pub struct Dimuon<Sc: Schedule = f64> {
    /// Transformation that clips the global norm of the gradients, or [`None`] without clipping.
    clipping: Option<ClipByGlobalNorm>,

    /// Transformation that orthogonalizes the conditioned momentum of the leaves.
    dimuon: ScaleByDimuon,

    /// Transformation that scales the updates by the negated learning rate.
    learning_rate: ScaleBySchedule<Sc>,
}

impl<Sc: Schedule> Dimuon<Sc> {
    /// Creates a new [`Dimuon`] with the learning rate schedule `learning_rate` (e.g., an `f64` for a constant learning
    /// rate) and default values for all other hyperparameters.
    pub fn new(learning_rate: Sc) -> Self {
        Self {
            clipping: None,
            dimuon: ScaleByDimuon::new(),
            learning_rate: ScaleBySchedule::learning_rate(learning_rate),
        }
    }

    /// Returns this optimizer with `beta1` as the decay `β₁` of its momentum.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `beta1` is not in `[0, 1)`.
    pub fn with_beta1(mut self, beta1: f64) -> Result<Self, OptimizerError> {
        self.dimuon = self.dimuon.with_beta1(beta1)?;
        Ok(self)
    }

    /// Returns this optimizer with `damping` as the relative damping `δ` of its row and column energies.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `damping` is negative or not finite.
    pub fn with_damping(mut self, damping: f64) -> Result<Self, OptimizerError> {
        self.dimuon = self.dimuon.with_damping(damping)?;
        Ok(self)
    }

    /// Returns this optimizer with `exponent` as the exponent `p` of the rank scaling `(r / 32)^p` of its updates, or
    /// without that factor when `exponent` is [`None`] (refer to [`ScaleByDimuon::with_rank_scaling_exponent`]).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `exponent` is not finite.
    pub fn with_rank_scaling_exponent(mut self, exponent: Option<f64>) -> Result<Self, OptimizerError> {
        self.dimuon = self.dimuon.with_rank_scaling_exponent(exponent)?;
        Ok(self)
    }

    /// Returns this optimizer with its gradients rescaled so that their global norm is at most `maximum_norm` before
    /// they enter the momentum, or without clipping when `maximum_norm` is [`None`] (which is what the nonpositive
    /// `grad_clip_norm` of the Tinker API selects).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `maximum_norm` is not positive and finite.
    pub fn with_gradient_clipping_norm(mut self, maximum_norm: Option<f64>) -> Result<Self, OptimizerError> {
        self.clipping = maximum_norm.map(ClipByGlobalNorm::new).transpose()?;
        Ok(self)
    }

    /// Returns the [`ClipByGlobalNorm`] transformation of this optimizer, if it clips its gradients.
    #[inline]
    pub fn clipping(&self) -> Option<&ClipByGlobalNorm> {
        self.clipping.as_ref()
    }

    /// Returns the [`ScaleByDimuon`] transformation of this optimizer.
    #[inline]
    pub fn dimuon(&self) -> &ScaleByDimuon {
        &self.dimuon
    }

    /// Returns the learning rate [`Schedule`] of this optimizer.
    #[inline]
    pub fn learning_rate(&self) -> &Sc {
        self.learning_rate.schedule()
    }
}

impl<P: OptimizerValue, Sc: Schedule> Optimizer<P> for Dimuon<Sc> {
    type State<S: Parameterized<P>> = DimuonState<P, S>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        Ok(DimuonState {
            dimuon: self.dimuon.initialize(parameters)?,
            learning_rate: self.learning_rate.initialize(parameters)?,
        })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        // The clipping comes first, so that the clipping scale multiplies the gradients that enter the momentum.
        let DimuonState { dimuon, learning_rate } = state;
        let gradients = match &self.clipping {
            Some(clipping) => clipping.update(gradients, (), parameters, arguments)?.0,
            None => gradients,
        };
        let (updates, dimuon) = self.dimuon.update(gradients, dimuon, parameters, arguments)?;
        let (updates, learning_rate) = self.learning_rate.update(updates, learning_rate, parameters, arguments)?;
        Ok((updates, DimuonState { dimuon, learning_rate }))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use pretty_assertions::assert_eq;

    use ryft_core::Array;

    use crate::tests::{assert_parity, assert_parity_with_tolerance, parity_updates};
    use crate::transformations::muon::tests::{
        REFERENCE_TOLERANCE, assert_close, assert_reference, iterate, matrix_norms, reference_leaf, reference_updates,
        standard,
    };

    use super::*;

    #[test]
    fn test_muon_new() {
        let optimizer = Muon::new(0.02);
        assert_eq!(optimizer.learning_rate(), &0.02);
        assert_eq!(optimizer.adam().learning_rate(), &0.02);
        assert!(optimizer.adam().adam().nesterov());
        assert_eq!(optimizer.adam().weight_decay().rate(), 0.0);
        assert_eq!(optimizer.consistent_root_mean_square(), None);
        assert_eq!(optimizer.weight_decay().rate(), 0.0);
        assert!(format!("{optimizer:?}").starts_with("Muon { muon: ScaleByMuon {"));
    }

    #[test]
    fn test_muon_with_hyperparameters() {
        let optimizer = Muon::new(0.02)
            .with_muon(ScaleByMuon::new().with_beta(0.9).unwrap())
            .with_adam(AdamW::new(0.01))
            .with_epsilon(1e-6)
            .unwrap()
            .with_nesterov(false)
            .with_first_moment_data_type(DataType::F32)
            .unwrap()
            .with_consistent_root_mean_square(Some(0.2))
            .unwrap()
            .with_weight_decay(0.1)
            .unwrap()
            .with_weight_decay_mask(|path| path.to_string().contains("weights"));
        assert_eq!(optimizer.muon().beta(), 0.9);
        assert_eq!((optimizer.muon().epsilon(), optimizer.adam().adam().epsilon()), (1e-6, 1e-6));
        assert!(!optimizer.muon().nesterov() && !optimizer.adam().adam().nesterov());
        assert_eq!(optimizer.adam().learning_rate(), &0.01);
        assert_eq!(optimizer.consistent_root_mean_square(), Some(0.2));
        assert_eq!(optimizer.weight_decay().rate(), 0.1);
        assert!(matches!(
            Muon::new(0.02).with_consistent_root_mean_square(Some(-1.0)),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`consistent_root_mean_square` must be finite and non-negative but got `-1`",
        ));
    }

    #[test]
    fn test_muon_initialize() {
        // The rank-2 leaf is updated with Muon and the scalar one with AdamW.
        let parameters = BTreeMap::from([
            ("bias", Array::scalar(1.0f32).unwrap()),
            ("weights", Array::matrix(2, 2, vec![1.0f32; 4]).unwrap()),
        ]);
        let state = Muon::new(0.02).initialize(&parameters).unwrap();
        assert_eq!(
            state.named_parameters().map(|(path, _)| path.to_string()).collect::<Vec<_>>(),
            vec![
                "$.muon[0].count",
                "$.muon[0].first_moments[$[\"weights\"]]",
                "$.muon[0].newton_schulz_coefficients",
                "$.learning_rate[0].count",
                "$.adam[0].adam.count",
                "$.adam[0].adam.first_moments[$[\"bias\"]]",
                "$.adam[0].adam.second_moments[$[\"bias\"]]",
                "$.adam[0].learning_rate.count",
            ],
        );

        // Without matrix leaves, there is no Muon state.
        let state = Muon::new(0.02).initialize(&vec![Array::scalar(1.0f32).unwrap()]).unwrap();
        assert!(state.muon().is_none() && state.learning_rate().is_none() && state.adam().is_some());
    }

    #[test]
    fn test_muon_update() {
        // The matrix leaf `diag(3, 4)` of shape `[2, 2]` (with the shape scaling `1`) gets `-η · (O + λ · θ)` with its
        // orthogonalization `O` (refer to `test_scale_by_muon_update`), the weight decay `λ = 0.1`, and the parameter
        // `θ = I`, and the scalar leaf gets the AdamW update of its gradient `2` with Nesterov momentum, which is
        // `-η · (0.9 · 0.1 / (1 - 0.9²) + 1)` at the first step.
        let optimizer = Muon::new(0.5).with_weight_decay(0.1).unwrap();
        let parameters = BTreeMap::from([
            ("bias", Array::scalar(0.0f32).unwrap()),
            ("weights", Array::matrix(2, 2, vec![1.0f32, 0.0, 0.0, 1.0]).unwrap()),
        ]);
        let gradients = BTreeMap::from([
            ("bias", Array::scalar(2.0f32).unwrap()),
            ("weights", Array::matrix(2, 2, vec![3.0f32, 0.0, 0.0, 4.0]).unwrap()),
        ]);
        let state = optimizer.initialize(&parameters).unwrap();
        let (updates, state) = optimizer.update(gradients, state, &parameters, &UpdateArguments::new()).unwrap();
        let (first, second) = (iterate(0.6, &standard()), iterate(0.8, &standard()));
        assert_close(&updates["weights"], &[-0.5 * (first + 0.1), 0.0, 0.0, -0.5 * (second + 0.1)]);
        assert_close(&updates["bias"], &[-0.5 * (0.9 * 0.1 / (1.0 - 0.81) + 1.0)]);
        assert_eq!(state.muon().unwrap().count(), &Array::scalar(1i32).unwrap());

        // With a consistent root mean square `0.2`, the shape scaling of a `[2, 3]` leaf is `0.2 · √3`, and without
        // it, the width transfer scaling of a `[2, 3]` leaf is `√(3 / 2)`.
        let parameters = vec![Array::matrix(2, 3, vec![0.0f32; 6]).unwrap()];
        let gradients = vec![Array::matrix(2, 3, vec![1.0f32, 2.0, 2.0, 2.0, 4.0, 4.0]).unwrap()];
        let scale = iterate(1.0, &standard()) / 45f64.sqrt();
        for (optimizer, shape_scale) in [
            (Muon::new(1.0), 1.5f64.sqrt()),
            (Muon::new(1.0).with_consistent_root_mean_square(Some(0.2)).unwrap(), 0.2 * 3f64.sqrt()),
        ] {
            let state = optimizer.initialize(&parameters).unwrap();
            let (updates, _) =
                optimizer.update(gradients.clone(), state, &parameters, &UpdateArguments::new()).unwrap();
            assert_close(&updates[0], &[1.0, 2.0, 2.0, 2.0, 4.0, 4.0].map(|element| -element * scale * shape_scale));
        }
    }

    #[test]
    fn test_muon_optax_parity() {
        // Optax 9ecde04b: `optax.contrib.muon(0.1)`. The `weights` matrix goes through the Newton–Schulz iterations
        // and the `bias` vector through Adam with Nesterov momentum.
        const DEFAULT: [[f32; 9]; 4] = [
            [
                -0.14736742,
                0.14736742,
                -0.14736743,
                -0.10965617,
                -0.02512444,
                0.00080090266,
                -0.0020538964,
                -0.066628136,
                0.070064195,
            ],
            [
                0.016486194,
                -0.06838632,
                -0.07850528,
                -0.064711206,
                -0.047827758,
                0.036884736,
                -0.05456478,
                -0.03575707,
                -0.106119834,
            ],
            [
                -0.10741883,
                0.024058696,
                0.01805392,
                -0.017869802,
                -0.029156351,
                -0.07682688,
                -0.042762954,
                -0.12385644,
                0.044690415,
            ],
            [
                -0.089282595,
                -0.03771009,
                0.05947834,
                -0.059738565,
                0.015993169,
                -0.1062261,
                0.041737765,
                -0.065377705,
                -0.06271963,
            ],
        ];
        assert_parity(&parity_updates(&Muon::new(0.1)), &DEFAULT);

        // Optax 9ecde04b: `optax.contrib.muon(0.1, weight_decay=0.01, nesterov=False, consistent_rms=0.2)`.
        const VARIANT: [[f32; 9]; 4] = [
            [
                -0.09999933,
                0.09999933,
                -0.09999933,
                -0.032015443,
                -0.0066062654,
                -2.3468236e-5,
                0.0014190618,
                -0.020345282,
                0.019067135,
            ],
            [
                -0.026633432,
                -0.016273059,
                -0.087105505,
                -0.032110404,
                -0.010542034,
                0.011060636,
                0.013126949,
                -0.018151617,
                0.012574104,
            ],
            [
                -0.06704981,
                0.007051367,
                -0.022722436,
                -0.020376509,
                -0.0115199825,
                -0.012306388,
                -0.0017682149,
                -0.0339641,
                0.014305088,
            ],
            [
                -0.07737498,
                -0.018474897,
                0.026137862,
                -0.029242624,
                -0.010512939,
                -0.021737698,
                0.0049295286,
                -0.026881883,
                -0.0065846415,
            ],
        ];
        let optimizer = Muon::new(0.1)
            .with_weight_decay(0.01)
            .unwrap()
            .with_nesterov(false)
            .with_consistent_root_mean_square(Some(0.2))
            .unwrap();
        assert_parity(&parity_updates(&optimizer), &VARIANT);
    }

    #[test]
    fn test_dimuon_new() {
        let optimizer = Dimuon::new(0.02);
        assert_eq!(optimizer.learning_rate(), &0.02);
        assert_eq!(optimizer.dimuon(), &ScaleByDimuon::new());
        assert_eq!(optimizer.clipping(), None);
        assert!(format!("{optimizer:?}").starts_with("Dimuon { clipping: None, dimuon: ScaleByDimuon {"));
    }

    #[test]
    fn test_dimuon_with_hyperparameters() {
        let optimizer = Dimuon::new(0.02)
            .with_beta1(0.8)
            .unwrap()
            .with_damping(0.01)
            .unwrap()
            .with_rank_scaling_exponent(None)
            .unwrap()
            .with_gradient_clipping_norm(Some(1.5))
            .unwrap();
        assert_eq!(
            optimizer.dimuon(),
            &ScaleByDimuon::new()
                .with_beta1(0.8)
                .unwrap()
                .with_damping(0.01)
                .unwrap()
                .with_rank_scaling_exponent(None)
                .unwrap(),
        );
        assert_eq!(optimizer.clipping(), Some(&ClipByGlobalNorm::new(1.5).unwrap()));
        assert_eq!(optimizer.with_gradient_clipping_norm(None).unwrap().clipping(), None);
        assert!(matches!(
            Dimuon::new(0.02).with_beta1(1.0),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`beta1` must be in `[0, 1)` but got `1`",
        ));
        assert!(matches!(
            Dimuon::new(0.02).with_damping(-1.0),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`damping` must be finite and non-negative but got `-1`",
        ));
        assert!(matches!(
            Dimuon::new(0.02).with_rank_scaling_exponent(Some(f64::INFINITY)),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`rank_scaling_exponent` must be finite but got `inf`",
        ));
        assert!(matches!(
            Dimuon::new(0.02).with_gradient_clipping_norm(Some(0.0)),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`maximum_norm` must be positive and finite but got `0`",
        ));
    }

    #[test]
    fn test_dimuon_initialize() {
        let parameters = BTreeMap::from([
            ("experts", reference_leaf(2, DataType::F32, &[1.0; 12])),
            ("weights", Array::matrix(3, 4, vec![1.0f32; 12]).unwrap()),
        ]);
        let state = Dimuon::new(0.02).with_gradient_clipping_norm(Some(1.0)).unwrap().initialize(&parameters).unwrap();
        assert_eq!(
            state.named_parameters().map(|(path, _)| path.to_string()).collect::<Vec<_>>(),
            vec!["$.dimuon.first_moments[\"experts\"]", "$.dimuon.first_moments[\"weights\"]", "$.learning_rate.count"],
        );
        assert_eq!(state.dimuon().first_moments()["weights"], Array::matrix(3, 4, vec![0.0f32; 12]).unwrap());
        assert_eq!(state.learning_rate().count(), &Array::scalar(0i32).unwrap());
    }

    #[test]
    fn test_dimuon_update() {
        // `dimuon_reference.py`: `Dimuon` with the learning rate `0.02`, `β₁ = 0.8`, and the clipping norm `1`, on
        // the three reference leaves together, so that the global norm covers all of them. The clipping is active in
        // all steps but the third, whose gradients are scaled by `0.05`.
        const DIMUON_UPDATES: [[[f32; 12]; 3]; 4] = [
            [
                [
                    -0.007202116,
                    -0.009939762,
                    -0.008002595,
                    -0.0023016538,
                    -0.0017211273,
                    0.0025674384,
                    0.0056484737,
                    0.0060729985,
                    0.010445504,
                    0.0051015867,
                    -0.0026416997,
                    -0.009142525,
                ],
                [
                    -0.0035615643,
                    -0.007038035,
                    -0.0072041024,
                    -0.0071287295,
                    0.0015775374,
                    0.009541931,
                    0.010759487,
                    0.005445054,
                    -0.0024302104,
                    -0.0037350662,
                    -0.007075327,
                    -0.007088192,
                ],
                [
                    -0.009915453,
                    -0.0071868775,
                    -0.0010782002,
                    -0.004373406,
                    0.0041672876,
                    0.010748042,
                    0.009244421,
                    0.00019390273,
                    -0.00894781,
                    -0.0048007644,
                    -0.007875603,
                    -0.0072464217,
                ],
            ],
            [
                [
                    -0.014171678,
                    -0.007636231,
                    -0.0008531793,
                    0.0028013096,
                    0.0047657033,
                    0.006925948,
                    0.003975218,
                    0.002157644,
                    0.0060019037,
                    -0.0053411694,
                    -0.006896192,
                    -0.007285435,
                ],
                [
                    -0.0072427117,
                    -0.0043595047,
                    -0.00039300477,
                    -0.0014239909,
                    0.007025616,
                    0.012266508,
                    0.009650528,
                    -0.00042238281,
                    -0.009234176,
                    -0.0073115607,
                    -0.0042989473,
                    -0.00023731038,
                ],
                [
                    -0.011066818,
                    -0.0014219234,
                    0.006285476,
                    0.0024941529,
                    0.007032431,
                    0.009135858,
                    0.0016851785,
                    -0.004966017,
                    -0.01367908,
                    -0.007580766,
                    -0.0054235756,
                    0.0012813105,
                ],
            ],
            [
                [
                    -0.012683462,
                    -0.005549727,
                    -0.0010890504,
                    0.0064897607,
                    0.005403313,
                    0.004853652,
                    0.00424872,
                    0.0028238178,
                    0.0023776968,
                    -0.0037158553,
                    -0.007027086,
                    -0.011893493,
                ],
                [
                    -0.007133562,
                    -0.004955624,
                    -0.00020512867,
                    0.0016728848,
                    0.0062190965,
                    0.00915547,
                    0.008251468,
                    -0.0017659523,
                    -0.0132668605,
                    -0.007206949,
                    -0.0049097566,
                    -3.330259e-5,
                ],
                [
                    -0.009355753,
                    -0.002466634,
                    0.008587569,
                    0.003753067,
                    0.0066594514,
                    0.008794816,
                    9.68494e-5,
                    -0.0073440773,
                    -0.008628694,
                    -0.01080218,
                    -0.0066879895,
                    0.003650729,
                ],
            ],
            [
                [
                    0.0050487965,
                    0.0066081746,
                    0.007467413,
                    0.008107507,
                    0.0027589086,
                    -0.0009155697,
                    -0.0037084734,
                    -0.009435855,
                    -0.010117673,
                    -0.004714758,
                    -0.00029106188,
                    0.009873042,
                ],
                [
                    9.267074e-5,
                    0.004579306,
                    0.008071777,
                    0.007916593,
                    0.00072020513,
                    -0.008781981,
                    -0.012276553,
                    -0.0074521555,
                    0.0022848062,
                    0.00026372724,
                    0.004671154,
                    0.008015214,
                ],
                [
                    0.008034059,
                    0.008298023,
                    0.00500732,
                    0.0051387316,
                    -0.0010384883,
                    -0.010825052,
                    -0.009325593,
                    -0.0031249863,
                    0.0047248327,
                    0.004456313,
                    0.010487317,
                    0.0073650875,
                ],
            ],
        ];
        let optimizer = Dimuon::new(0.02).with_beta1(0.8).unwrap().with_gradient_clipping_norm(Some(1.0)).unwrap();
        let updates = reference_updates(&optimizer, DataType::F32);
        let precise = reference_updates(&optimizer, DataType::F64);
        assert_reference(&precise, &DIMUON_UPDATES, 1e-6);
        assert_reference(&updates, &DIMUON_UPDATES, REFERENCE_TOLERANCE);

        // Every matrix of every update has the Frobenius norm `η · √r · (r / 32)^0.175`.
        let norm = |rank: f64| 0.02 * rank.sqrt() * (rank / 32.0).powf(0.175);
        for step in &updates {
            let norms = [matrix_norms(&step[0], 1), matrix_norms(&step[1], 1), matrix_norms(&step[2], 2)].concat();
            let expected = [norm(3.0), norm(3.0), norm(2.0), norm(2.0)];
            assert!(norms.iter().zip(expected).all(|(norm, expected)| (norm - expected).abs() < 1e-7), "{norms:?}");
        }
    }

    #[test]
    fn test_dimuon_update_clipping() {
        // The gradient `diag(3, 4)` has the global norm `5`, so the clipping norm `1` scales it by `1 / 5` before it
        // enters the momentum `0.1 · g`, while the gradient `diag(0.3, 0.4)` of norm `0.5` enters unchanged. The first
        // update is the same in all cases, because the normalization removes the scale of the momentum.
        let parameters = vec![Array::matrix(2, 2, vec![0.0f32; 4]).unwrap()];
        let clipped = Dimuon::new(0.1).with_gradient_clipping_norm(Some(1.0)).unwrap();
        let unclipped = Dimuon::new(0.1);
        let large = vec![Array::matrix(2, 2, vec![3.0f32, 0.0, 0.0, 4.0]).unwrap()];
        let small = vec![Array::matrix(2, 2, vec![0.3f32, 0.0, 0.0, 0.4]).unwrap()];
        let step = |optimizer: &Dimuon, gradients: &Vec<Array>| {
            let state = optimizer.initialize(&parameters).unwrap();
            optimizer.update(gradients.clone(), state, &parameters, &UpdateArguments::new()).unwrap()
        };
        let (clipped_updates, clipped_state) = step(&clipped, &large);
        let (unclipped_updates, unclipped_state) = step(&unclipped, &large);
        let (small_updates, small_state) = step(&clipped, &small);
        let moments =
            |state: &DimuonState<Array, Vec<Array>>| state.dimuon().first_moments()[0].elements::<f32>().unwrap();
        assert_parity_with_tolerance(&[moments(&clipped_state)], &[[0.06f32, 0.0, 0.0, 0.08]], 1e-6);
        assert_parity_with_tolerance(&[moments(&unclipped_state)], &[[0.3f32, 0.0, 0.0, 0.4]], 1e-6);
        assert_parity_with_tolerance(&[moments(&small_state)], &[[0.03f32, 0.0, 0.0, 0.04]], 1e-6);
        let expected = [unclipped_updates[0].elements::<f32>().unwrap()];
        assert_parity_with_tolerance(&[clipped_updates[0].elements::<f32>().unwrap()], &expected, 1e-6);
        assert_parity_with_tolerance(&[small_updates[0].elements::<f32>().unwrap()], &expected, 1e-6);
    }
}
