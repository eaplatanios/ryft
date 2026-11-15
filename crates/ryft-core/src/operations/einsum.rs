//! Einstein summation through [`Einsum`] and [`einsum`]. An einsum contracts any number of arrays according to
//! subscripts such as `"ij,jk->ik"`, which name every axis of every input and of the output. Labels that are absent
//! from the output are summed over, labels that are shared between inputs are aligned, and a label that is repeated
//! within one input selects its diagonal.
//!
//! Einsum is not a primitive operation. Like `jax.numpy.einsum`, it is decomposed into the pairwise [`Dot`]
//! contractions, [`Reduce`] sums, [`Diagonal`]s, and [`Transpose`]s of an [`EinsumPlan`], so every program transform
//! and backend supports it through those operations. The module has three layers:
//!
//!   - **Specification:** [`EinsumSpecification`] holds parsed subscripts, including ellipses and an implicit output.
//!   - **Plan:** [`EinsumSpecification::plan`] expands the subscripts for particular input types, selects the order
//!     of the pairwise contractions with an [`EinsumPathStrategy`], and returns the [`EinsumPlan`] that
//!     [`EinsumPlan::contract`] executes or stages. [`EinsumPlan::contract_with`] replaces the pairwise
//!     contraction (e.g., with a quantized dot operation).
//!   - **Frontends:** The [`Einsum`] value capability plans and contracts inline on every call, and the global
//!     [`einsum`] function creates an [`EinsumComposite`], which plans once for each input type signature and calls
//!     the contraction as one named composite.
//!
//! The path search is a port of [`opt_einsum`](https://github.com/dgasmith/opt_einsum) 3.4.0, which `jax.numpy.einsum`
//! uses, so that [`EinsumPathStrategy::Auto`] selects the same path as JAX for the same inputs and the staged
//! contractions match JAX's.
//!
//! # Example
//!
//! ```rust
//! # use ryft_core::{Array, Einsum, ProgramError};
//! # fn main() -> Result<(), ProgramError> {
//! let x = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
//! let y = Array::matrix(3, 2, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0])?;
//! assert_eq!(Array::einsum("ij,jk->ik", [&x, &y])?, Array::matrix(2, 2, vec![4.0, 5.0, 10.0, 11.0])?);
//! assert_eq!(x.einsum_with("ij,jk->", [&y])?, Array::scalar(30.0)?);
//! # Ok(())
//! # }
//! ```

use std::cell::Cell;
use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, VecDeque};
use std::fmt::Display;
use std::str::FromStr;
use std::sync::{Arc, Mutex};

use ryft_macros::capability;

use crate::arrays::{ArrayIrType, ArrayType, DataType, Dimension, Sharding};
use crate::contexts::{Context, Domain, StagingContext};
use crate::operations::Capability;
use crate::operations::collectives::parallel_vary::{
    ManualVariationAlignment, ParallelVaryOperation, manual_variation_axes, vary_over_missing_manual_axes,
};
use crate::operations::composites::builders::{Composite, CompositeCallContext, composite};
use crate::operations::composites::configurations::JvpFromPrimal;
use crate::operations::composites::rules::definitions::{CompositeAttribute, CompositeKey};
use crate::operations::dot::{
    Dot, DotDimensionNumbers, DotOperation, DotPrecision, is_dot_accumulation_type, is_native_f8_dot_type,
};
use crate::operations::manipulation::conversions::{ConvertElementType, ConvertElementTypeOperation};
use crate::operations::manipulation::diagonal::{Diagonal, diagonal_type};
use crate::operations::manipulation::reshaping::Reshape;
use crate::operations::manipulation::transposition::Transpose;
use crate::operations::reductions::{Reduce, ReduceOperation, ReductionKind};
use crate::programs::{Operation, ProgramError, ProvenanceScope, TypeError, Typed, Value, ValueProjection};
use crate::tracing::{DomainTracer, DomainTracingContext};

/// Name that diagnostics and provenance scopes of this module use for an einsum operation.
const EINSUM_NAME: &str = "einsum";

// TODO(eaplatanios): Review from here onwards.

/// One element of the subscript of an einsum input or output.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EinsumLabel {
    /// Label of one axis. Axes with the same label are aligned across inputs, and labels are ordered by their
    /// characters wherever an order is needed (e.g., for the axes of an implicit output).
    Named(char),

    /// Ellipsis (i.e., `...`), which stands for all axes of its input that no label names. The ellipses of all inputs
    /// are aligned at their last axes.
    Ellipsis,
}

impl EinsumLabel {
    /// Returns the label that the integer `index` of a label list denotes, or [`None`] if there are not that many
    /// labels. Indices map to `a` through `z` and then `A` through `Z`, like the consecutive integer labels that
    /// `jax.numpy.einsum` accepts in place of subscripts, and then to the alphabetic Unicode characters from `À` on in
    /// the order of their code points, so that every returned label is valid in an [`EinsumSpecification`].
    pub fn from_index(index: usize) -> Option<Self> {
        const LETTERS: &[u8; 52] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let label = match LETTERS.get(index) {
            Some(letter) => Some(char::from(*letter)),
            None => ('À'..=char::MAX).filter(|label| label.is_alphabetic()).nth(index - LETTERS.len()),
        };
        label.map(Self::Named)
    }
}

impl Display for EinsumLabel {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Named(label) => write!(formatter, "{label}"),
            Self::Ellipsis => formatter.write_str("..."),
        }
    }
}

impl From<char> for EinsumLabel {
    #[inline]
    fn from(label: char) -> Self {
        Self::Named(label)
    }
}

/// Parsed subscripts of an einsum, which name the axes of every input and, optionally, of the output. The
/// [`FromStr`] implementation parses the string form of NumPy's
/// [`einsum`](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html) (e.g., `"ij,jk->ik"` or
/// `"...ij,...jk"`), and [`Self::from_labels`] builds a specification from label lists.
///
///   - **Inputs:** Each input subscript has one label for every axis of its input. At most one ellipsis may stand for
///     the axes that no label names. A label that appears more than once in one input selects the diagonal along
///     its axes.
///   - **Output:** An explicit output lists every output axis once, in order, and every one of its labels must
///     appear in an input. Labels that it omits are summed over. Without an output (i.e., the implicit form), the
///     output consists of the axes of the ellipses followed by every label that appears exactly once across all
///     inputs, in the order of their characters.
///
/// A specification is independent of input types. [`Self::plan`] expands its ellipses and validates it against
/// particular inputs.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct EinsumSpecification {
    /// Subscript of each input.
    inputs: Vec<Vec<EinsumLabel>>,

    /// Subscript of the output, or [`None`] for the implicit form.
    output: Option<Vec<EinsumLabel>>,
}

impl EinsumSpecification {
    /// Creates a new [`EinsumSpecification`] from the label lists of its inputs and, optionally, of its output.
    ///
    /// # Errors
    ///
    /// Returns a [`TypeError`] if there is no input, a label is not alphabetic, a subscript contains more than one
    /// ellipsis, the output repeats a label, or the output contains a label that no input contains.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ryft_core::{EinsumLabel, EinsumSpecification};
    /// let labels = |indices: &[usize]| indices.iter().flat_map(|index| EinsumLabel::from_index(*index)).collect();
    /// let inputs = vec![labels(&[0, 1]), labels(&[1, 2])];
    /// assert_eq!(EinsumSpecification::from_labels(inputs, Some(labels(&[0, 2]))), "ab,bc->ac".parse());
    /// ```
    pub fn from_labels(inputs: Vec<Vec<EinsumLabel>>, output: Option<Vec<EinsumLabel>>) -> Result<Self, TypeError> {
        if inputs.is_empty() {
            return Err(TypeError::invalid(format!("`{EINSUM_NAME}` subscripts must have at least one input")));
        }
        for subscript in inputs.iter().chain(&output) {
            for label in subscript {
                if let EinsumLabel::Named(label) = label
                    && !label.is_alphabetic()
                {
                    return Err(TypeError::invalid(format!(
                        "`{EINSUM_NAME}` subscripts contain the invalid label `{label}`",
                    )));
                }
            }
            if subscript.iter().filter(|label| **label == EinsumLabel::Ellipsis).count() > 1 {
                return Err(TypeError::invalid(format!(
                    "`{EINSUM_NAME}` subscripts contain more than one ellipsis in one subscript",
                )));
            }
        }
        if let Some(output) = &output {
            for (position, label) in output.iter().enumerate() {
                let EinsumLabel::Named(label) = label else { continue };
                if output[..position].contains(&EinsumLabel::Named(*label)) {
                    return Err(TypeError::invalid(format!(
                        "`{EINSUM_NAME}` output label `{label}` appears more than once",
                    )));
                }
                if !inputs.iter().any(|input| input.contains(&EinsumLabel::Named(*label))) {
                    return Err(TypeError::invalid(format!(
                        "`{EINSUM_NAME}` output label `{label}` does not appear in any input",
                    )));
                }
            }
        }
        Ok(Self { inputs, output })
    }

    /// Returns the subscript of each input of this [`EinsumSpecification`].
    #[inline]
    pub fn inputs(&self) -> &[Vec<EinsumLabel>] {
        &self.inputs
    }

    /// Returns the subscript of the output of this [`EinsumSpecification`], or [`None`] for the implicit form.
    #[inline]
    pub fn output(&self) -> Option<&[EinsumLabel]> {
        self.output.as_deref()
    }

    /// Plans this einsum for inputs of the provided types. This expands the ellipses for the ranks of the inputs,
    /// validates the subscripts against the input types, and selects the order of the pairwise contractions with the
    /// [`EinsumPathStrategy`] of `options`. This is the analogue of `jax.numpy.einsum_path`, and the returned
    /// [`EinsumPlan`] can be executed or staged any number of times.
    ///
    /// The inputs must have the same element type, because an einsum converts no input implicitly. The exception is
    /// that inputs may have distinct 8-bit floating-point types when `options` request an accumulation type, like
    /// the inputs of a [`DotOperation`], which contracts such pairs natively. Each label must
    /// have one extent across all inputs that contain it, except that inputs may have extent one where others have
    /// another extent (static or dynamic), in which case they are broadcast. The axes of a label that one input
    /// repeats must have equal extents. Dynamic extents take part in the path search as extent 8, like in JAX.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] if the number of inputs or their ranks do not match the subscripts, their element
    /// types differ, the extents of a label are inconsistent, or an explicit path is invalid for the inputs.
    pub fn plan(&self, input_types: &[ArrayType], options: &EinsumOptions) -> Result<EinsumPlan, ProgramError> {
        if input_types.len() != self.inputs.len() {
            return Err(TypeError::invalid(format!(
                "`{}` subscripts `{}` have {} inputs but got {}",
                EINSUM_NAME,
                self,
                self.inputs.len(),
                input_types.len(),
            ))
            .into());
        }
        let data_type = input_types[0].data_type();
        let are_native_f8_types = options.accumulation_type.is_some()
            && input_types.iter().all(|input_type| is_native_f8_dot_type(input_type.data_type()));
        if let Some(other) = input_types.iter().map(ArrayType::data_type).find(|other| *other != data_type)
            && !are_native_f8_types
        {
            return Err(TypeError::invalid(format!(
                "`{EINSUM_NAME}` inputs must have the same element type but got `{data_type}` and `{other}`",
            ))
            .into());
        }
        let ranks = input_types.iter().map(ArrayType::rank).collect::<Vec<_>>();
        let (input_names, output_names) = self.expand(&ranks)?;

        // Every label has one extent across the inputs that contain it. An input may have extent one where another
        // has a different extent, in which case its axis is squeezed away before the two are contracted, but the
        // axes of a label that one input repeats must agree, because their diagonal is taken. The path search uses
        // the largest extent of every label and treats a dynamic extent as extent 8, like JAX does.
        let mut extents = BTreeMap::<char, Dimension>::new();
        for (index, (names, input_type)) in input_names.iter().zip(input_types).enumerate() {
            let mut input_extents = BTreeMap::<char, Dimension>::new();
            for (axis, name) in names.iter().enumerate() {
                let extent = input_type.dimension(axis);
                if input_extents.get(name).is_some_and(|other| *other != extent) {
                    return Err(TypeError::invalid(format!(
                        "`{EINSUM_NAME}` label `{name}` is repeated in input {index} of type `{input_type}` along \
                         axes with different extents",
                    ))
                    .into());
                }
                input_extents.insert(*name, extent.clone());
                match extents.get(name) {
                    Some(other) if *other == extent || extent == Dimension::Static(1) => {}
                    Some(other) if *other != Dimension::Static(1) => {
                        return Err(TypeError::invalid(format!(
                            "`{EINSUM_NAME}` label `{name}` has extent `{extent}` in input {index} but extent \
                             `{other}` in an earlier input",
                        ))
                        .into());
                    }
                    _ => {
                        extents.insert(*name, extent);
                    }
                }
            }
        }
        let sizes = extents
            .iter()
            .map(|(name, extent)| (*name, extent.value().map_or(8, |extent| extent as u128)))
            .collect::<BTreeMap<_, _>>();

        // A single input needs no contraction, and two inputs have only one. More inputs are contracted pairwise in
        // the order that the strategy selects, where each pair of positions refers to the list of the remaining
        // inputs followed by the results of the earlier contractions.
        let input_sets = input_names.iter().map(|names| names.iter().copied().collect()).collect::<Vec<LabelSet>>();
        let output_set = output_names.iter().copied().collect::<LabelSet>();
        let input_count = input_names.len();
        let path = match options.path() {
            EinsumPathStrategy::Explicit(path) => {
                let mut remaining = input_count;
                for (first, second) in path {
                    if first == second || *first >= remaining || *second >= remaining {
                        return Err(TypeError::invalid(format!(
                            "`{EINSUM_NAME}` path contracts the positions {first} and {second}, which must be two \
                             distinct positions among its {remaining} remaining inputs",
                        ))
                        .into());
                    }
                    remaining -= 1;
                }
                if remaining != 1 {
                    return Err(TypeError::invalid(format!(
                        "`{}` path must have {} pairwise contractions for {} inputs but has {}",
                        EINSUM_NAME,
                        input_count - 1,
                        input_count,
                        path.len(),
                    ))
                    .into());
                }
                path.clone()
            }
            _ if input_count == 1 => Vec::new(),
            _ if input_count == 2 => vec![(0, 1)],
            EinsumPathStrategy::Unoptimized => vec![(0, 1); input_count - 1],
            EinsumPathStrategy::Optimal => ssa_to_linear(&optimal_ssa_path(&input_sets, &output_set, &sizes)),
            EinsumPathStrategy::Greedy => ssa_to_linear(&greedy_ssa_path(&input_sets, &output_set, &sizes)),
            EinsumPathStrategy::Auto => ssa_to_linear(&match input_count {
                ..=4 => optimal_ssa_path(&input_sets, &output_set, &sizes),
                5..=6 => branch_ssa_path(&input_sets, &output_set, &sizes, None),
                7..=8 => branch_ssa_path(&input_sets, &output_set, &sizes, Some(2)),
                9..=14 => branch_ssa_path(&input_sets, &output_set, &sizes, Some(1)),
                _ => greedy_ssa_path(&input_sets, &output_set, &sizes),
            }),
        };

        // Each step contracts one or two of the remaining inputs, removes the labels that neither the output nor any
        // other remaining input needs, and appends its result to the remaining inputs. The positions are taken from
        // the right, so the input at the larger position is the left-hand side of a pairwise step. An intermediate
        // result orders its labels by their first appearance in its two inputs, which minimizes transposes, and the
        // last step produces the output order. This is the contraction list of `opt_einsum.contract_path`.
        let mut remaining_sets = input_sets;
        let mut remaining_names = input_names.clone();
        let positions = if input_count == 1 {
            vec![vec![0]]
        } else {
            path.iter().map(|(first, second)| vec![*first.max(second), *first.min(second)]).collect()
        };
        let step_count = positions.len();
        let mut steps = Vec::with_capacity(step_count);
        for (index, positions) in positions.into_iter().enumerate() {
            let step_sets = positions.iter().map(|position| remaining_sets.remove(*position)).collect::<Vec<_>>();
            let step_names = positions.iter().map(|position| remaining_names.remove(*position)).collect::<Vec<_>>();
            let contracted_set = step_sets.iter().flatten().copied().collect::<LabelSet>();
            let needed_set = remaining_sets.iter().flatten().chain(&output_set).copied().collect::<LabelSet>();
            let result_set = contracted_set.intersection(&needed_set).copied().collect::<LabelSet>();
            let result_names = if index + 1 == step_count {
                output_names.clone()
            } else {
                let appearance = step_names.iter().flatten().copied().collect::<Vec<_>>();
                let mut result_names = result_set.iter().copied().collect::<Vec<_>>();
                result_names.sort_by_key(|name| appearance.iter().position(|other| other == name));
                result_names
            };
            steps.push(EinsumStep {
                positions,
                contracted_names: contracted_set.difference(&result_set).copied().collect(),
                input_names: step_names,
                result_names: result_names.clone(),
            });
            remaining_sets.push(result_set);
            remaining_names.push(result_names);
        }

        Ok(EinsumPlan {
            specification: self.clone(),
            options: options.clone(),
            input_types: input_types.to_vec(),
            input_labels: input_names,
            output_labels: output_names,
            path,
            steps,
        })
    }

    /// Expands the ellipses of this specification for inputs of the provided ranks and resolves an implicit output,
    /// returning the labels of every input axis and of every output axis. The axes of the ellipses are named with the
    /// first symbols of [`einsum_symbol`] that the subscripts do not use, and the ellipses of all inputs are aligned
    /// at their last axes. This follows `opt_einsum.parser.parse_einsum_input`, so that the characters of the labels,
    /// which determine every order that an einsum derives from labels, are those that JAX uses.
    fn expand(&self, ranks: &[usize]) -> Result<(Vec<Vec<char>>, Vec<char>), TypeError> {
        let has_ellipsis =
            self.inputs.iter().chain(&self.output).flatten().any(|label| *label == EinsumLabel::Ellipsis);
        let used = self
            .inputs
            .iter()
            .chain(&self.output)
            .flatten()
            .filter_map(|label| match label {
                EinsumLabel::Named(label) => Some(*label),
                EinsumLabel::Ellipsis => None,
            })
            .collect::<BTreeSet<_>>();
        let ellipsis_names = (0..)
            .map(einsum_symbol)
            .filter(|symbol| !used.contains(symbol))
            .take(if has_ellipsis { ranks.iter().copied().max().unwrap_or(0) } else { 0 })
            .collect::<Vec<_>>();

        let mut longest = 0;
        let mut input_names = Vec::with_capacity(self.inputs.len());
        for (index, (subscript, rank)) in self.inputs.iter().zip(ranks).enumerate() {
            let named_count = subscript.iter().filter(|label| **label != EinsumLabel::Ellipsis).count();
            let has_ellipsis = named_count != subscript.len();
            if (has_ellipsis && named_count > *rank) || (!has_ellipsis && named_count != *rank) {
                return Err(TypeError::invalid(format!(
                    "`{EINSUM_NAME}` subscript {index} of `{self}` does not match the rank {rank} of its input",
                )));
            }
            let ellipsis_count = rank - named_count;
            longest = longest.max(ellipsis_count);
            input_names.push(
                subscript
                    .iter()
                    .flat_map(|label| match label {
                        EinsumLabel::Named(label) => vec![*label],
                        EinsumLabel::Ellipsis => ellipsis_names[ellipsis_names.len() - ellipsis_count..].to_vec(),
                    })
                    .collect::<Vec<_>>(),
            );
        }

        let output_ellipsis_names = &ellipsis_names[ellipsis_names.len() - longest..];
        let output_names = match &self.output {
            Some(output) => output
                .iter()
                .flat_map(|label| match label {
                    EinsumLabel::Named(label) => vec![*label],
                    EinsumLabel::Ellipsis => output_ellipsis_names.to_vec(),
                })
                .collect::<Vec<_>>(),
            None => {
                // The implicit output has the axes of the ellipses first and then every other label that appears
                // exactly once across all inputs, in the order of their characters.
                let mut counts = BTreeMap::<char, usize>::new();
                for name in input_names.iter().flatten() {
                    *counts.entry(*name).or_default() += 1;
                }
                output_ellipsis_names
                    .iter()
                    .copied()
                    .chain(
                        counts
                            .into_iter()
                            .filter(|(name, count)| *count == 1 && !output_ellipsis_names.contains(name))
                            .map(|(name, _)| name),
                    )
                    .collect()
            }
        };
        Ok((input_names, output_names))
    }
}

impl Display for EinsumSpecification {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (index, input) in self.inputs.iter().enumerate() {
            if index > 0 {
                formatter.write_str(",")?;
            }
            for label in input {
                write!(formatter, "{label}")?;
            }
        }
        if let Some(output) = &self.output {
            formatter.write_str("->")?;
            for label in output {
                write!(formatter, "{label}")?;
            }
        }
        Ok(())
    }
}

impl FromStr for EinsumSpecification {
    type Err = TypeError;

    fn from_str(subscripts: &str) -> Result<Self, TypeError> {
        // Whitespace is insignificant, a single `->` separates the inputs from an explicit output, commas separate
        // the inputs, and three consecutive periods form an ellipsis.
        let subscripts = subscripts.chars().filter(|character| !character.is_whitespace()).collect::<String>();
        let parse = |subscript: &str| {
            let mut labels = Vec::new();
            let mut characters = subscript.chars().peekable();
            while let Some(character) = characters.next() {
                if character != '.' {
                    labels.push(EinsumLabel::Named(character));
                } else if characters.next() == Some('.') && characters.next() == Some('.') {
                    labels.push(EinsumLabel::Ellipsis);
                } else {
                    return Err(TypeError::invalid(format!(
                        "`{EINSUM_NAME}` subscripts contain an invalid ellipsis in `{subscript}`",
                    )));
                }
            }
            Ok(labels)
        };
        let (inputs, output) = match subscripts.split_once("->") {
            Some((inputs, output)) => (inputs, Some(output)),
            None => (subscripts.as_str(), None),
        };
        if inputs.contains(['-', '>']) || output.is_some_and(|output| output.contains(['-', '>', ','])) {
            return Err(TypeError::invalid(format!(
                "`{EINSUM_NAME}` subscripts must contain at most one `->`, after all of the inputs",
            )));
        }
        Self::from_labels(inputs.split(',').map(parse).collect::<Result<Vec<_>, _>>()?, output.map(parse).transpose()?)
    }
}

/// Strategy that selects the order of the pairwise contractions of an einsum with more than two inputs. The order
/// does not change the result, up to floating-point rounding, but it determines the sizes of the intermediate results
/// and so the cost of the einsum. The searches are ports of the path optimizers of
/// [`opt_einsum`](https://dgasmith.github.io/opt_einsum/paths/introduction/) with the same names, and they select the
/// same paths.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum EinsumPathStrategy {
    /// Selects the search by the number of inputs, aiming to keep it fast. Up to four inputs use
    /// [`Self::Optimal`], up to fourteen use branching searches that explore only the most promising contractions
    /// (all of them up to six inputs, two up to eight, and one beyond), and more inputs use [`Self::Greedy`]. This is
    /// the default and the `"auto"` strategy that `jax.numpy.einsum` uses by default.
    #[default]
    Auto,

    /// Explores the orders exhaustively like the `optimal` optimizer of `opt_einsum`, whose choice it reproduces,
    /// aiming for the fewest scalar operations. Its cost grows factorially with the number of inputs.
    Optimal,

    /// Repeatedly contracts the pair of inputs that reduces the total size of the remaining inputs the most.
    Greedy,

    /// Repeatedly contracts the first two of the remaining inputs and appends their result to the remaining inputs,
    /// like `optimize=False` in JAX. Four inputs are therefore contracted as two pairs whose results are contracted
    /// last, and not from left to right.
    Unoptimized,

    /// Uses the provided pairwise contractions. Each pair holds two positions in the list of the remaining inputs,
    /// from which a contraction removes its two inputs and to which it appends its result. This is the form of the
    /// paths that [`EinsumPlan::path`] returns.
    Explicit(Vec<(usize, usize)>),
}

/// Options of an einsum, which select how its contractions are ordered and computed.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct EinsumOptions {
    /// Strategy that selects the order of the pairwise contractions.
    path: EinsumPathStrategy,

    /// Optional element type at which the einsum accumulates and which its output has.
    accumulation_type: Option<DataType>,

    /// Precision of every pairwise contraction.
    precision: DotPrecision,

    /// Optional requested [`Sharding`] of the output.
    output_sharding: Option<Sharding>,
}

impl EinsumOptions {
    /// Creates new [`EinsumOptions`] with the [`EinsumPathStrategy::Auto`] strategy, no accumulation type, the default
    /// precision, and no requested output sharding.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns these options with the provided [`EinsumPathStrategy`].
    #[inline]
    pub fn with_path(mut self, path: EinsumPathStrategy) -> Self {
        self.path = path;
        self
    }

    /// Returns these options with the provided accumulation type, which is the element type of the output. Every
    /// pairwise contraction accumulates at it (refer to [`DotOperation::with_accumulation_type`]), and the output is
    /// converted to it last. Values are converted to it earlier where JAX converts them. A sum converts its input
    /// first, unless the element type of the input is the type that the two promote to (e.g., `bf16` inputs are
    /// summed at an `f16` accumulation type and `f32` inputs are summed at `f32`). The 8-bit floating-point types
    /// promote to no other type and are converted as well, where JAX fails. A contraction converts its inputs first
    /// when a `dot` cannot accumulate them at that type itself, which it can for floating-point inputs at every
    /// floating-point type and for integers at wider integers of the same signedness. That covers integers at any
    /// numeric type that is at least as wide (e.g., `u8` at `i32` and `i32` at `f32`) and every other element type
    /// that promotes to the accumulation type. The remaining combinations are rejected (e.g., `i32` at `i8`). This is
    /// the `preferred_element_type` argument of `jax.numpy.einsum`.
    #[inline]
    pub fn with_accumulation_type<D: Into<Option<DataType>>>(mut self, accumulation_type: D) -> Self {
        self.accumulation_type = accumulation_type.into();
        self
    }

    /// Returns these options with the provided [`DotPrecision`], which every pairwise contraction uses (refer to
    /// [`DotOperation::with_precision`]). This is the `precision` argument of `jax.numpy.einsum`.
    #[inline]
    pub fn with_precision<P: Into<DotPrecision>>(mut self, precision: P) -> Self {
        self.precision = precision.into();
        self
    }

    /// Returns these options with a requested output [`Sharding`], which the last pairwise contraction receives (refer
    /// to [`DotOperation::with_output_sharding`]), reordered to the order of its axes when a transpose follows it.
    /// An einsum whose last step is not a pairwise contraction (i.e., an einsum of one input) rejects the request.
    /// The sharding must cover every axis of the output, and the results of the earlier contractions infer their
    /// shardings from their inputs. This is the `out_sharding` argument of `jax.numpy.einsum`, which additionally
    /// accepts a sharding of fewer axes and stages its earlier contractions with automatic sharding.
    #[inline]
    pub fn with_output_sharding<S: Into<Option<Sharding>>>(mut self, output_sharding: S) -> Self {
        self.output_sharding = output_sharding.into();
        self
    }

    /// Returns the [`EinsumPathStrategy`] of these options.
    #[inline]
    pub fn path(&self) -> &EinsumPathStrategy {
        &self.path
    }

    /// Returns the optional accumulation type of these options.
    #[inline]
    pub fn accumulation_type(&self) -> Option<DataType> {
        self.accumulation_type
    }

    /// Returns the [`DotPrecision`] of these options.
    #[inline]
    pub fn precision(&self) -> DotPrecision {
        self.precision
    }

    /// Returns the requested output [`Sharding`] of these options, if any.
    #[inline]
    pub fn output_sharding(&self) -> Option<&Sharding> {
        self.output_sharding.as_ref()
    }
}

/// Plan of an einsum for inputs of particular types, which [`EinsumSpecification::plan`] creates. It holds the labels
/// of every input and output axis after expanding ellipses, the selected order of the pairwise contractions, and the
/// steps that realize it.
///
/// Every step consumes one or two of the remaining inputs and appends its result to them, following
/// `jax.numpy.einsum` step for step, so that equal paths stage the same contractions:
///
///   1. Axes of extent one whose label has a larger extent in the other input are squeezed away.
///   2. Labels that only one input contains once and that no later step needs are summed out of that input.
///   3. Labels that one input repeats are replaced by their [`Diagonal`], which reads only the elements of the
///      diagonal, where JAX masks the whole input and sums it. A diagonal that no later step needs is summed.
///   4. The two inputs are contracted with one [`Dot`], whose batching dimensions are the shared labels that are still
///      needed and whose contracting dimensions are the shared labels that are not, in the order of their characters.
///      The contraction is first tried with its inputs exchanged, which more often produces the needed order of axes.
///   5. The result is transposed when its axes are still not in the needed order.
///
/// A plan can be reused for other inputs with the same ranks and consistent label extents, although its path was
/// selected for the extents of the types that it was created for.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct EinsumPlan {
    /// Specification that this plan was created from.
    specification: EinsumSpecification,

    /// Options that this plan was created with.
    options: EinsumOptions,

    /// Input types that this plan was created for.
    input_types: Vec<ArrayType>,

    /// Labels of the axes of every input, with ellipses expanded.
    input_labels: Vec<Vec<char>>,

    /// Labels of the axes of the output, with ellipses expanded.
    output_labels: Vec<char>,

    /// Selected pairwise contractions. Refer to [`EinsumPathStrategy::Explicit`] for their form.
    path: Vec<(usize, usize)>,

    /// Steps that realize the path.
    steps: Vec<EinsumStep>,
}

impl EinsumPlan {
    /// Returns the [`EinsumSpecification`] that this plan was created from.
    #[inline]
    pub fn specification(&self) -> &EinsumSpecification {
        &self.specification
    }

    /// Returns the [`EinsumOptions`] that this plan was created with.
    #[inline]
    pub fn options(&self) -> &EinsumOptions {
        &self.options
    }

    /// Returns the labels of the axes of every input, with ellipses expanded.
    #[inline]
    pub fn input_labels(&self) -> &[Vec<char>] {
        &self.input_labels
    }

    /// Returns the labels of the axes of the output, with ellipses expanded and an implicit output resolved.
    #[inline]
    pub fn output_labels(&self) -> &[char] {
        &self.output_labels
    }

    /// Returns the selected pairwise contractions, which is empty for an einsum of one input. Each pair holds two
    /// positions in the list of the remaining inputs, from which a contraction removes its two inputs and to which it
    /// appends its result. The path can be passed to [`EinsumPathStrategy::Explicit`].
    #[inline]
    pub fn path(&self) -> &[(usize, usize)] {
        &self.path
    }

    /// Returns the type of the output that [`Self::contract`] produces for inputs of the types that this plan was
    /// created for, by inferring the types of all of its steps without executing or staging them.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] if a step is invalid for its input types (e.g., a pairwise contraction whose
    /// [`DotPrecision`] does not support them, or a requested output sharding that it rejects).
    pub fn output_type(&self) -> Result<ArrayType, ProgramError> {
        Ok(self.infer()?.0)
    }

    /// Executes or stages this plan on `inputs` with [`Dot`] as the pairwise contraction, which uses the precision,
    /// accumulation type, and requested output sharding of the [`EinsumOptions`] of this plan. Concrete values compute
    /// the result and values that carry a context stage the steps of this plan inline.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] if the number of inputs or their ranks do not match this plan, or if a step fails.
    pub fn contract<V>(&self, inputs: &[V]) -> Result<V, ProgramError>
    where
        V: Value<Type = ArrayType> + ConvertElementType + Diagonal + Dot + Reduce + Reshape + Transpose,
    {
        self.contract_with_main_diagonal(inputs, |value: &V, kept_axis, removed_axis| {
            value.diagonal(0, kept_axis, removed_axis)
        })
    }

    /// Executes or stages this plan on `inputs` like [`Self::contract`], but with `main_diagonal` as the function
    /// that takes the main diagonal of a value along a kept axis and a removed axis, in place of the kept axis.
    fn contract_with_main_diagonal<V, F>(&self, inputs: &[V], main_diagonal: F) -> Result<V, ProgramError>
    where
        V: Value<Type = ArrayType> + ConvertElementType + Dot + Reduce + Reshape + Transpose,
        F: FnMut(&V, usize, usize) -> Result<V, ProgramError>,
    {
        let precision = self.options.precision;
        let accumulation_type = self.options.accumulation_type;
        let mut steps = ValueSteps {
            contraction: |lhs: &V, rhs: &V, dimensions: &DotDimensionNumbers, output_sharding: Option<&Sharding>| {
                lhs.dot_with_configuration(rhs, dimensions, precision, accumulation_type, output_sharding)
            },
            main_diagonal,
        };
        self.execute(inputs, accumulation_type, &mut steps)
    }

    /// Executes or stages this plan on `inputs` like [`Self::contract`], but with `contraction` as the pairwise
    /// contraction. This is the analogue of the experimental `_dot_general` argument of `jax.numpy.einsum`, and it
    /// lets an einsum use another bilinear contraction (e.g., a quantized dot). The precision and the accumulation
    /// type of the [`EinsumOptions`] of this plan are not used, because `contraction` selects its own, and no
    /// element type is converted.
    ///
    /// # Parameters
    ///
    ///   - `inputs`: Inputs of the einsum, in the order of its subscripts.
    ///   - `contraction`: Function that contracts its left-hand side and right-hand side inputs with the provided
    ///     [`DotDimensionNumbers`], whose result must have the axes of [`DotOperation`] in the same order. It also
    ///     receives the requested output [`Sharding`] when it computes the last contraction and one was requested.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] if the number of inputs or their ranks do not match this plan, or if a step fails.
    pub fn contract_with<V, F>(&self, inputs: &[V], contraction: F) -> Result<V, ProgramError>
    where
        V: Value<Type = ArrayType> + ConvertElementType + Diagonal + Reduce + Reshape + Transpose,
        F: FnMut(&V, &V, &DotDimensionNumbers, Option<&Sharding>) -> Result<V, ProgramError>,
    {
        let main_diagonal = |value: &V, kept_axis, removed_axis| value.diagonal(0, kept_axis, removed_axis);
        self.execute(inputs, None, &mut ValueSteps { contraction, main_diagonal })
    }

    /// Infers the type of the output of [`Self::contract`] for the input types of this plan, together with the number
    /// of instructions that it stages, by running the steps of this plan over types.
    fn infer(&self) -> Result<(ArrayType, usize), ProgramError> {
        let accumulation_type = self.options.accumulation_type;
        let mut steps = TypeSteps { precision: self.options.precision, accumulation_type, instruction_count: 0 };
        let output_type = self.execute(&self.input_types, accumulation_type, &mut steps)?;
        Ok((output_type, steps.instruction_count))
    }

    /// Runs the steps of this plan on `inputs` through `steps`, which either computes or stages every step on values
    /// or infers its type. This is a port of the `_einsum` function of JAX, except for repeated labels, whose
    /// diagonals are taken with [`Diagonal`].
    fn execute<V: Clone, S: EinsumSteps<V>>(
        &self,
        inputs: &[V],
        accumulation_type: Option<DataType>,
        steps: &mut S,
    ) -> Result<V, ProgramError> {
        if inputs.len() != self.input_labels.len() {
            return Err(TypeError::invalid(format!(
                "`{}` subscripts `{}` have {} inputs but got {}",
                EINSUM_NAME,
                self.specification,
                self.input_labels.len(),
                inputs.len(),
            ))
            .into());
        }
        for (index, (input, names)) in inputs.iter().zip(&self.input_labels).enumerate() {
            let rank = steps.array_type(input).rank();
            if rank != names.len() {
                return Err(TypeError::invalid(format!(
                    "`{}` subscript {} of `{}` does not match the rank {} of its input",
                    EINSUM_NAME, index, self.specification, rank,
                ))
                .into());
            }
        }

        let mut state = EinsumExecution { steps, accumulation_type };
        let mut values = inputs.to_vec();
        for (index, step) in self.steps.iter().enumerate() {
            let is_last = index + 1 == self.steps.len();
            let output_sharding = if is_last { self.options.output_sharding.as_ref() } else { None };
            let (value, names) = match step.positions.as_slice() {
                [position] => {
                    if output_sharding.is_some() {
                        return Err(TypeError::invalid(format!(
                            "`{EINSUM_NAME}` output sharding requires a pairwise contraction as the last step, which \
                             an einsum of one input does not have",
                        ))
                        .into());
                    }

                    // Labels that appear once and are not in the output are summed with a single reduction, and
                    // repeated labels are replaced by their diagonal, which is summed when it is not in the output.
                    let value = values.remove(*position);
                    let names = step.input_names[0].clone();
                    let counts = count_names(&names);
                    let uniques =
                        step.contracted_names.iter().copied().filter(|name| count_of(&counts, name) == 1).collect();
                    let (value, names) = state.sum_uniques(value, names, uniques)?;
                    state.sum_repeats(value, names, &counts, &step.result_names)?
                }
                [lhs_position, rhs_position] => {
                    let lhs = values.remove(*lhs_position);
                    let rhs = values.remove(*rhs_position);
                    let lhs_names = step.input_names[0].clone();
                    let rhs_names = step.input_names[1].clone();

                    // An axis of extent one is squeezed away when the other input has another extent for its label,
                    // which broadcasts the squeezed input along that label.
                    let rhs_type = state.steps.array_type(&rhs);
                    let (lhs, lhs_names) = state.filter_singleton_axes(lhs, lhs_names, &rhs_type, &rhs_names)?;
                    let lhs_type = state.steps.array_type(&lhs);
                    let (rhs, rhs_names) = state.filter_singleton_axes(rhs, rhs_names, &lhs_type, &lhs_names)?;

                    // Labels that only one input contains, once, and that the result does not need are summed out of
                    // that input before the contraction.
                    let lhs_counts = count_names(&lhs_names);
                    let rhs_counts = count_names(&rhs_names);
                    let lhs_uniques = step
                        .contracted_names
                        .iter()
                        .copied()
                        .filter(|name| count_of(&lhs_counts, name) == 1 && count_of(&rhs_counts, name) == 0)
                        .collect();
                    let (lhs, lhs_names) = state.sum_uniques(lhs, lhs_names, lhs_uniques)?;
                    let rhs_uniques = step
                        .contracted_names
                        .iter()
                        .copied()
                        .filter(|name| count_of(&rhs_counts, name) == 1 && count_of(&lhs_counts, name) == 0)
                        .collect();
                    let (rhs, rhs_names) = state.sum_uniques(rhs, rhs_names, rhs_uniques)?;

                    // Labels that one input repeats are replaced by their diagonal, which is kept when the result or
                    // the other input contains the label and summed otherwise.
                    let keep_names = step.result_names.iter().chain(&rhs_names).copied().collect::<Vec<_>>();
                    let (lhs, lhs_names) = state.sum_repeats(lhs, lhs_names, &lhs_counts, &keep_names)?;
                    let keep_names = step.result_names.iter().chain(&lhs_names).copied().collect::<Vec<_>>();
                    let (rhs, rhs_names) = state.sum_repeats(rhs, rhs_names, &rhs_counts, &keep_names)?;

                    // The shared labels that the result needs are batching dimensions, in the order of the result,
                    // and the remaining shared labels are contracting dimensions, in the order of their characters.
                    let position = |names: &[char], name: &char| names.iter().position(|other| other == name).unwrap();
                    let contracted_names = step
                        .contracted_names
                        .iter()
                        .filter(|name| lhs_names.contains(name) || rhs_names.contains(name))
                        .collect::<Vec<_>>();
                    let batch_names = step
                        .result_names
                        .iter()
                        .filter(|name| lhs_names.contains(name) && rhs_names.contains(name))
                        .collect::<Vec<_>>();
                    let lhs_batch = batch_names.iter().map(|name| position(&lhs_names, name)).collect::<Vec<_>>();
                    let rhs_batch = batch_names.iter().map(|name| position(&rhs_names, name)).collect::<Vec<_>>();
                    let lhs_contracting =
                        contracted_names.iter().map(|name| position(&lhs_names, name)).collect::<Vec<_>>();
                    let rhs_contracting =
                        contracted_names.iter().map(|name| position(&rhs_names, name)).collect::<Vec<_>>();
                    let remaining = |names: &[char]| {
                        names
                            .iter()
                            .filter(|name| !batch_names.contains(name) && !contracted_names.contains(name))
                            .copied()
                            .collect::<Vec<_>>()
                    };
                    let remaining_lhs_names = remaining(&lhs_names);
                    let remaining_rhs_names = remaining(&rhs_names);
                    let batch_names = batch_names.into_iter().copied().collect::<Vec<_>>();

                    // Both orders of the two inputs are tried, in the hope that one of them needs no transpose. The
                    // path search tends to contract from right to left, so the exchanged order is tried first.
                    let exchanged_names = [batch_names.as_slice(), &remaining_rhs_names, &remaining_lhs_names].concat();
                    if exchanged_names == step.result_names {
                        let dimensions =
                            DotDimensionNumbers::new(rhs_contracting, lhs_contracting, rhs_batch, lhs_batch);
                        (state.contract(&rhs, &lhs, &dimensions, output_sharding)?, exchanged_names)
                    } else {
                        // A requested output sharding describes the axes of the output, so the contraction receives
                        // it in the order of its own axes when a transpose follows.
                        let names = [batch_names.as_slice(), &remaining_lhs_names, &remaining_rhs_names].concat();
                        let output_sharding = match output_sharding {
                            Some(output_sharding) if names != step.result_names => Some(output_sharding.transpose(
                                names.iter().map(|name| position(&step.result_names, name)).collect::<Vec<_>>(),
                            )?),
                            output_sharding => output_sharding.cloned(),
                        };
                        let dimensions =
                            DotDimensionNumbers::new(lhs_contracting, rhs_contracting, lhs_batch, rhs_batch);
                        (state.contract(&lhs, &rhs, &dimensions, output_sharding.as_ref())?, names)
                    }
                }
                _ => unreachable!("every step of an einsum plan contracts one or two inputs"),
            };

            // The result has the labels of the step in some order, which a transpose turns into the needed one.
            let value = if names == step.result_names {
                value
            } else {
                let permutation = step
                    .result_names
                    .iter()
                    .map(|name| names.iter().position(|other| other == name).unwrap())
                    .collect::<Vec<_>>();
                state.steps.transpose(&value, permutation)?
            };
            values.push(value);
        }

        let output = values.remove(0);
        match accumulation_type {
            Some(accumulation_type) => state.convert(output, accumulation_type),
            None => Ok(output),
        }
    }
}

/// Value-level Einstein summation capability, which contracts any number of values according to subscripts such as
/// `"ij,jk->ik"`. Every function plans the einsum for the types of its inputs and then executes or stages the
/// [`EinsumPlan`] inline, like `jax.numpy.einsum` does on every call. Refer to [`EinsumSpecification`] for the
/// subscripts, to [`EinsumSpecification::plan`] for the requirements on the inputs, and to [`EinsumPlan`] for the
/// decomposition. Use the global [`einsum`] function instead to plan once and call the contraction as one unit.
///
/// The universe parameter `T` defaults to the [`Capability`] universe of the implementor, so that homogeneous array
/// values implement this capability for [`ArrayType`] and array IR values implement it for [`ArrayIrType`]. Array IR
/// values contract through their array views and take the diagonals of repeated labels themselves, so their dynamic
/// extents are supported, including along a label that one input repeats when its axes share one extent.
///
/// # Example
///
/// ```rust
/// # use ryft_core::{Array, Einsum, ProgramError};
/// # fn main() -> Result<(), ProgramError> {
/// let matrix = Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0])?;
/// let vector = Array::vector(vec![10.0, 100.0])?;
/// assert_eq!(Array::einsum("ij,j->i", [&matrix, &vector])?, Array::vector(vec![210.0, 430.0])?);
/// assert_eq!(matrix.einsum_with("ii->i", [])?, Array::vector(vec![1.0, 4.0])?);
/// assert_eq!(matrix.einsum_with("ij,kj", [&matrix])?, Array::matrix(2, 2, vec![5.0, 11.0, 11.0, 25.0])?);
/// # Ok(())
/// # }
/// ```
#[capability]
pub trait Einsum<T = <Self as Capability>::Universe>: Capability + Sized {
    /// Contracts `inputs` according to `subscripts`, which are parsed into an [`EinsumSpecification`], with the
    /// default [`EinsumOptions`].
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] if `subscripts` are invalid or do not match `inputs`, or if a step of the einsum
    /// cannot be executed or staged.
    #[inline]
    fn einsum<'i, I: IntoIterator<Item = &'i Self>>(subscripts: &str, inputs: I) -> Result<Self, ProgramError>
    where
        Self: 'i,
    {
        Self::einsum_with_options(&subscripts.parse()?, inputs, &EinsumOptions::default())
    }

    /// Contracts `self` followed by `others` according to `subscripts`, like [`Self::einsum`].
    #[inline]
    fn einsum_with<'i, I: IntoIterator<Item = &'i Self>>(
        &'i self,
        subscripts: &str,
        others: I,
    ) -> Result<Self, ProgramError> {
        Self::einsum(subscripts, std::iter::once(self).chain(others))
    }

    /// Contracts `inputs` according to `specification` with the provided [`EinsumOptions`], which select the order of
    /// the pairwise contractions, their precision, the accumulation type, and a requested output sharding. This is
    /// the most general function of this capability, and the other functions are implemented using it.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] if `specification` does not match `inputs` (refer to
    /// [`EinsumSpecification::plan`]), or if a step of the einsum cannot be executed or staged.
    fn einsum_with_options<'i, I: IntoIterator<Item = &'i Self>>(
        specification: &EinsumSpecification,
        inputs: I,
        options: &EinsumOptions,
    ) -> Result<Self, ProgramError>
    where
        Self: 'i;
}

impl<V> Einsum<ArrayType> for V
where
    V: Value<Type = ArrayType, Domain: Context> + ConvertElementType + Diagonal + Dot + Reduce + Reshape + Transpose,
{
    fn einsum_with_options<'i, I: IntoIterator<Item = &'i Self>>(
        specification: &EinsumSpecification,
        inputs: I,
        options: &EinsumOptions,
    ) -> Result<Self, ProgramError>
    where
        Self: 'i,
    {
        let inputs = inputs.into_iter().cloned().collect::<Vec<_>>();
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let plan = specification.plan(&input_types, options)?;
        invoke_in_scope(ProvenanceScope::new(specification.to_string()), &inputs, || plan.contract(&inputs))
    }
}

impl<V: Value<Type = ArrayIrType> + ValueProjection<ArrayType> + Diagonal<ArrayIrType>> Einsum<ArrayIrType> for V
where
    V::Projected: Value<Type = ArrayType, Domain: Context> + ConvertElementType + Dot + Reduce + Reshape + Transpose,
{
    fn einsum_with_options<'i, I: IntoIterator<Item = &'i Self>>(
        specification: &EinsumSpecification,
        inputs: I,
        options: &EinsumOptions,
    ) -> Result<Self, ProgramError>
    where
        Self: 'i,
    {
        // The einsum runs in the array universe over the array views of the inputs, which rejects first-class
        // dimension members. Only the diagonals of repeated labels are taken on the array IR values themselves,
        // whose `Diagonal` supports dynamic extents.
        let inputs = inputs
            .into_iter()
            .map(|input| <V as ValueProjection<ArrayType>>::into_projected(input.clone()))
            .collect::<Result<Vec<_>, _>>()?;
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let plan = specification.plan(&input_types, options)?;
        let scope = ProvenanceScope::new(specification.to_string());
        let output = invoke_in_scope(scope, &inputs, || {
            plan.contract_with_main_diagonal(&inputs, |value: &V::Projected, kept_axis, removed_axis| {
                let value = <V as ValueProjection<ArrayType>>::from_projected(value.clone());
                Ok(<V as ValueProjection<ArrayType>>::into_projected(value.diagonal(0, kept_axis, removed_axis)?)?)
            })
        })?;
        Ok(<V as ValueProjection<ArrayType>>::from_projected(output))
    }
}

/// Name of the [`CompositeKey`]s of the named composites that [`einsum`] creates.
pub const EINSUM_COMPOSITE_NAME: &str = "ryft.einsum";

/// Version of the [`CompositeKey`]s of the named composites that [`einsum`] creates. It changes whenever the
/// computation that a key describes changes.
pub const EINSUM_COMPOSITE_VERSION: u32 = 1;

/// Einstein summation function created by [`einsum`] and [`einsum_with_options`], which calls the contraction as one
/// named composite. The function plans its einsum once for each input type signature and keeps the
/// [`EinsumPlan`]s, where the [`Einsum`] capability plans on every call, and each of its calls is one unit whose body
/// is the decomposition of the plan:
///
///   - A staged program holds one `composite` instruction for each call, and the calls with equal input types share
///     one body. A backend lowers the body, sharing one function between repeated calls.
///   - A backend that executes eagerly runs a whole call as one compiled unit, where the inline capability dispatches
///     every step of the decomposition separately.
///   - Differentiation and batching are derived from the body, so no rule is written by hand. Forward and reverse
///     mode differentiate the body in the caller, and batching batches the body structurally.
///
/// A plan that stages at most one instruction (e.g., a matrix product) gains nothing from the composite, so such
/// calls contract inline. Create the function once and reuse it, because a new function plans and traces its body
/// again and shares no compiled unit with another one. Use the [`Einsum`] capability instead for a custom pairwise
/// contraction or for a requested output sharding, neither of which the key of a named composite can describe.
///
/// # Performance
///
/// The function pays off where dispatch dominates the contractions. The table shows warm times for each call on
/// eager XLA, measured on 2026-11-14 on one CPU device for `f32` matrices of extent 8 (in batches of 4 for the
/// `b` label), with the outputs of 20,000 calls awaited together.
///
///   | Subscripts        | Instructions | [`Einsum`] capability | [`EinsumComposite`] |
///   |-------------------|--------------|-----------------------|---------------------|
///   | `ij,jk->ik`       | 1            | 7.0 µs                | 5.7 µs              |
///   | `bij,bjk->ikb`    | 2            | 9.9 µs                | 6.0 µs              |
///   | `ij,jk,kl->il`    | 2            | 14.8 µs               | 6.3 µs              |
///   | `ij,jk,kl,lm->im` | 3            | 26.0 µs               | 6.9 µs              |
///
/// The capability parses and plans on every call and dispatches every instruction, where a call of the function is
/// one dispatch after a lookup of its plan. The two forms take the same time once the contractions dominate (at
/// extent 256 in the same setup). Awaiting every output before the next call adds several microseconds to each call
/// of either form, and it made the capability the faster one for `bij,bjk->ikb` (12 µs against 17 µs), so measure
/// when the latency of single calls matters.
pub struct EinsumComposite<D: Domain<Type = ArrayType>>
where
    DomainTracingContext<D>: StagingContext<Type = ArrayType, Constant = D::Constant, Operation = D::Operation>,
{
    /// Plans of this einsum for the input type signatures that it was called with.
    plans: Arc<EinsumPlans>,

    /// Named composite whose body contracts its inputs with the plan for their types.
    composite: Composite<D::Constant, D::Operation, Vec<DomainTracer<D>>, DomainTracer<D>, JvpFromPrimal>,
}

impl<D: Domain<Type = ArrayType, Constant: 'static, Operation: 'static>> EinsumComposite<D>
where
    DomainTracingContext<D>: StagingContext<Type = ArrayType, Constant = D::Constant, Operation = D::Operation>,
{
    /// Returns the [`EinsumSpecification`] of this function.
    #[inline]
    pub fn specification(&self) -> &EinsumSpecification {
        &self.plans.specification
    }

    /// Returns the [`EinsumOptions`] of this function.
    #[inline]
    pub fn options(&self) -> &EinsumOptions {
        &self.plans.options
    }

    /// Contracts `inputs` according to the specification and the options of this function. Refer to
    /// [`EinsumSpecification::plan`] for the requirements on the inputs. Manual variation is aligned before the
    /// composite boundary, so its transpose sums the contributions of replicated inputs once.
    ///
    /// The domain of the inputs supplies the call context, so this function can be reused with eager and transformed
    /// values that share its constant and operation types.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] if the specification does not match `inputs`, or if a step of the einsum cannot be
    /// executed or staged.
    pub fn call<C, V>(&self, inputs: Vec<V>) -> Result<V, ProgramError>
    where
        C: CompositeCallContext<Type = ArrayType, Value = V, Constant = D::Constant, Operation = D::Operation>,
        V: Value<Type = ArrayType, Domain = C>
            + ManualVariationAlignment
            + ConvertElementType
            + Diagonal
            + Dot
            + Reduce
            + Reshape
            + Transpose,
    {
        let inputs = V::align_manual_variation(&inputs)?;
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let planned = self.plans.planned(&input_types)?;
        if planned.instruction_count <= 1 {
            // One instruction is cheaper to stage or dispatch directly than through a composite boundary.
            return invoke_in_scope(planned.scope.clone(), &inputs, || planned.plan.contract(&inputs));
        }
        self.composite.call(inputs)
    }
}

/// Creates an Einstein summation function for `subscripts` with the default [`EinsumOptions`]. Refer to
/// [`einsum_with_options`] for more information.
///
/// # Example
///
/// ```rust
/// # use ryft_core::{Array, ArrayOperation, EagerContext, Einsum, differentiate_at, einsum};
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let chain = einsum::<EagerContext<Array, ArrayOperation<Array>>>("ij,jk,kl->il")?;
/// let x = Array::matrix(1, 2, vec![1.0, 2.0])?;
/// let y = Array::matrix(2, 2, vec![1.0, 0.0, 0.0, 1.0])?;
/// let z = Array::matrix(2, 1, vec![3.0, 4.0])?;
/// assert_eq!(chain.call(vec![x.clone(), y.clone(), z.clone()])?, Array::matrix(1, 1, vec![11.0])?);
///
/// // The gradient with respect to the first input is derived from the decomposition.
/// let gradient = differentiate_at(x)
///     .with_captures((y, z))
///     .gradient(|x, (y, z)| chain.call(vec![x, y, z])?.einsum_with("ij->", []))?;
/// assert_eq!(gradient, Array::matrix(1, 2, vec![3.0, 4.0])?);
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns a [`ProgramError`] if `subscripts` cannot be parsed into an [`EinsumSpecification`].
pub fn einsum<D>(subscripts: &str) -> Result<EinsumComposite<D>, ProgramError>
where
    D: Domain<Type = ArrayType, Constant: 'static, Operation: 'static>,
    DomainTracer<D>: ConvertElementType + Diagonal + Dot + Reduce + Reshape + Transpose,
    DomainTracingContext<D>: StagingContext<Type = ArrayType, Constant = D::Constant, Operation = D::Operation>,
{
    einsum_with_options(subscripts.parse()?, EinsumOptions::default())
}

/// Creates an Einstein summation function for `specification` with the provided [`EinsumOptions`], which calls the
/// contraction as one named composite. Refer to [`EinsumComposite`] for what the returned function does and for when
/// to prefer it over the [`Einsum`] capability, and to [`EinsumSpecification`] for the subscripts.
///
/// The [`CompositeKey`] of the composite has the name [`EINSUM_COMPOSITE_NAME`] and the version
/// [`EINSUM_COMPOSITE_VERSION`], and its attributes encode the subscripts, the path strategy, the accumulation type,
/// and the precision, so that equal keys describe equal computations.
///
/// # Errors
///
/// Returns a [`ProgramError`] if `options` request an output sharding, which the key of a named composite cannot
/// describe.
pub fn einsum_with_options<D>(
    specification: EinsumSpecification,
    options: EinsumOptions,
) -> Result<EinsumComposite<D>, ProgramError>
where
    D: Domain<Type = ArrayType, Constant: 'static, Operation: 'static>,
    DomainTracer<D>: ConvertElementType + Diagonal + Dot + Reduce + Reshape + Transpose,
    DomainTracingContext<D>: StagingContext<Type = ArrayType, Constant = D::Constant, Operation = D::Operation>,
{
    if options.output_sharding.is_some() {
        return Err(ProgramError::UnsupportedOperation {
            message: format!(
                "`{EINSUM_NAME}` composites do not support a requested output sharding; use the `Einsum` capability \
                 instead",
            ),
        });
    }
    let key = einsum_composite_key(&specification, &options);
    let plans = Arc::new(EinsumPlans { specification, options, planned: Mutex::new(VecDeque::new()) });
    let body_plans = plans.clone();
    let rule_plans = plans.clone();
    let composite = composite(move |inputs: Vec<DomainTracer<D>>| {
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        body_plans.planned(&input_types)?.plan.contract(&inputs)
    })
    .with_named_composite(key, move |input_types: Vec<ArrayType>| {
        Ok(rule_plans.planned(&input_types)?.output_type.clone())
    })
    .with_jvp_from_primal();
    Ok(EinsumComposite { plans, composite })
}

/// Invokes `contract`, which contracts `inputs` with the plan of an einsum, inside the provided [`ProvenanceScope`],
/// which is named after the specification of the einsum. The scope is purely diagnostic. It attributes every
/// instruction that the decomposition stages to its einsum, like the named call that JAX wraps its einsum in, and it
/// is a no-op under an eager context.
fn invoke_in_scope<V: Value<Domain: Context>, R, F: FnOnce() -> R>(
    scope: ProvenanceScope,
    inputs: &[V],
    contract: F,
) -> R {
    match inputs.first() {
        Some(input) => input.domain().invoke_with_provenance_scope(scope, contract),
        None => contract(),
    }
}

/// Set of labels of an einsum.
type LabelSet = BTreeSet<char>;

/// Plans of one einsum for the input type signatures that an [`EinsumComposite`] was called with, which the output
/// type rule and the body of its composite and its calls share, so that each signature is planned once.
struct EinsumPlans {
    /// Specification of the einsum.
    specification: EinsumSpecification,

    /// Options of the einsum.
    options: EinsumOptions,

    /// Most recently planned input type signatures with their plans, oldest first.
    planned: Mutex<VecDeque<(Vec<ArrayType>, Arc<PlannedEinsum>)>>,
}

impl EinsumPlans {
    /// Number of input type signatures whose plans are kept, which matches the number of signatures for which a
    /// composite keeps its traced bodies.
    const CAPACITY: usize = 16;

    /// Returns the plan for inputs of the provided types, planning the einsum if it has no plan for them yet. The
    /// lock is not held while planning, whose path search can be expensive, so concurrent calls with new input types
    /// may each plan them, in which case the first plan is kept.
    fn planned(&self, input_types: &[ArrayType]) -> Result<Arc<PlannedEinsum>, ProgramError> {
        let find = |planned: &VecDeque<(Vec<ArrayType>, Arc<PlannedEinsum>)>| {
            planned.iter().find(|(planned_types, _)| planned_types == input_types).map(|(_, plan)| plan.clone())
        };
        if let Some(plan) = find(&self.planned.lock().expect("einsum plans should not be poisoned")) {
            return Ok(plan);
        }
        let plan = self.specification.plan(input_types, &self.options)?;
        let (output_type, instruction_count) = plan.infer()?;
        let scope = ProvenanceScope::new(self.specification.to_string());
        let plan = Arc::new(PlannedEinsum { plan, output_type, instruction_count, scope });
        let mut planned = self.planned.lock().expect("einsum plans should not be poisoned");
        if let Some(plan) = find(&planned) {
            return Ok(plan);
        }
        if planned.len() == Self::CAPACITY {
            planned.pop_front();
        }
        planned.push_back((input_types.to_vec(), plan.clone()));
        Ok(plan)
    }
}

/// [`EinsumPlan`] together with what [`EinsumPlan::infer`] computes for it.
struct PlannedEinsum {
    /// Plan of the einsum.
    plan: EinsumPlan,

    /// Type of the output of the plan.
    output_type: ArrayType,

    /// Number of instructions that the plan stages.
    instruction_count: usize,

    /// Provenance scope of the instructions that the plan stages inline, which is named after the specification.
    scope: ProvenanceScope,
}

/// One step of an [`EinsumPlan`], which contracts one or two of the remaining inputs.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct EinsumStep {
    /// Positions of the inputs of this step in the list of the remaining inputs. Each position is taken from the list
    /// after removing the inputs at the positions before it, and a pairwise step holds the larger position first.
    positions: Vec<usize>,

    /// Labels that this step removes, in the order of their characters.
    contracted_names: Vec<char>,

    /// Labels of the axes of every input of this step, in the order of [`Self::positions`].
    input_names: Vec<Vec<char>>,

    /// Labels of the axes of the result of this step.
    result_names: Vec<char>,
}

/// Primitive steps of an einsum over values of type `V`, which [`EinsumPlan::execute`] is generic over, so that one
/// implementation of the decomposition both computes or stages values and infers types.
trait EinsumSteps<V> {
    /// Returns the type of `value`.
    fn array_type(&self, value: &V) -> ArrayType;

    /// Removes the provided axes of `value`, which have extent one.
    fn squeeze(&mut self, value: &V, axes: Vec<usize>) -> Result<V, ProgramError>;

    /// Sums `value` along the provided axes.
    fn reduce_sum(&mut self, value: &V, axes: &[usize]) -> Result<V, ProgramError>;

    /// Returns the diagonal of `value` along `kept_axis` and `removed_axis`, in place of `kept_axis`.
    fn diagonal(&mut self, value: &V, kept_axis: usize, removed_axis: usize) -> Result<V, ProgramError>;

    /// Contracts `lhs` and `rhs` with the provided dimension numbers, optionally requesting an output sharding.
    fn contract(
        &mut self,
        lhs: &V,
        rhs: &V,
        dimensions: &DotDimensionNumbers,
        output_sharding: Option<&Sharding>,
    ) -> Result<V, ProgramError>;

    /// Transposes `value` with the provided permutation.
    fn transpose(&mut self, value: &V, permutation: Vec<usize>) -> Result<V, ProgramError>;

    /// Converts `value` to the provided element type.
    fn convert_element_type(&mut self, value: &V, data_type: DataType) -> Result<V, ProgramError>;
}

/// [`EinsumSteps`] that compute or stage every step on values through their capabilities, with `Contraction` as the
/// pairwise contraction and `MainDiagonal` as the function that takes the main diagonal of two axes.
struct ValueSteps<Contraction, MainDiagonal> {
    /// Pairwise contraction.
    contraction: Contraction,

    /// Function that returns the main diagonal of a value along a kept axis and a removed axis, in place of the kept
    /// axis. Array IR values supply their own, which supports dynamic extents.
    main_diagonal: MainDiagonal,
}

impl<V, Contraction, MainDiagonal> EinsumSteps<V> for ValueSteps<Contraction, MainDiagonal>
where
    V: Value<Type = ArrayType> + ConvertElementType + Reduce + Reshape + Transpose,
    Contraction: FnMut(&V, &V, &DotDimensionNumbers, Option<&Sharding>) -> Result<V, ProgramError>,
    MainDiagonal: FnMut(&V, usize, usize) -> Result<V, ProgramError>,
{
    #[inline]
    fn array_type(&self, value: &V) -> ArrayType {
        value.r#type().into_owned()
    }

    #[inline]
    fn squeeze(&mut self, value: &V, axes: Vec<usize>) -> Result<V, ProgramError> {
        value.squeeze(axes)
    }

    #[inline]
    fn reduce_sum(&mut self, value: &V, axes: &[usize]) -> Result<V, ProgramError> {
        value.reduce_sum(axes, None)
    }

    #[inline]
    fn diagonal(&mut self, value: &V, kept_axis: usize, removed_axis: usize) -> Result<V, ProgramError> {
        (self.main_diagonal)(value, kept_axis, removed_axis)
    }

    #[inline]
    fn contract(
        &mut self,
        lhs: &V,
        rhs: &V,
        dimensions: &DotDimensionNumbers,
        output_sharding: Option<&Sharding>,
    ) -> Result<V, ProgramError> {
        (self.contraction)(lhs, rhs, dimensions, output_sharding)
    }

    #[inline]
    fn transpose(&mut self, value: &V, permutation: Vec<usize>) -> Result<V, ProgramError> {
        value.transpose(permutation)
    }

    #[inline]
    fn convert_element_type(&mut self, value: &V, data_type: DataType) -> Result<V, ProgramError> {
        value.convert_element_type(data_type)
    }
}

/// [`EinsumSteps`] that infer the type of every step through the type inference of the operation that
/// [`ValueSteps`] stages for it, counting the instructions that the steps stage.
struct TypeSteps {
    /// Precision of every pairwise contraction.
    precision: DotPrecision,

    /// Optional accumulation type of every pairwise contraction.
    accumulation_type: Option<DataType>,

    /// Number of instructions that the steps inferred so far stage.
    instruction_count: usize,
}

impl EinsumSteps<ArrayType> for TypeSteps {
    #[inline]
    fn array_type(&self, value: &ArrayType) -> ArrayType {
        value.clone()
    }

    fn squeeze(&mut self, value: &ArrayType, axes: Vec<usize>) -> Result<ArrayType, ProgramError> {
        self.instruction_count += 1;
        value.squeeze(axes)
    }

    fn reduce_sum(&mut self, value: &ArrayType, axes: &[usize]) -> Result<ArrayType, ProgramError> {
        self.instruction_count += 1;
        let operation = ReduceOperation::new(axes.to_vec(), ReductionKind::Sum);
        Ok(operation.infer_output_types(std::slice::from_ref(value), &[])?.remove(0))
    }

    fn diagonal(
        &mut self,
        value: &ArrayType,
        kept_axis: usize,
        removed_axis: usize,
    ) -> Result<ArrayType, ProgramError> {
        // The main diagonal of two axes that share one extent is an iota and a gather.
        self.instruction_count += 2;
        diagonal_type(value, 0, kept_axis, removed_axis)
    }

    fn contract(
        &mut self,
        lhs: &ArrayType,
        rhs: &ArrayType,
        dimensions: &DotDimensionNumbers,
        output_sharding: Option<&Sharding>,
    ) -> Result<ArrayType, ProgramError> {
        // The `Dot` capability of staged values aligns the manual variation of its inputs before it stages their
        // contraction (refer to `ManualVariationAlignment`). Each input is placed on the mesh of a manual axis that
        // it lacks, if it has no sharding yet, and is then marked as varying over that axis.
        let types = [lhs.clone(), rhs.clone()];
        let instruction_count = Cell::new(self.instruction_count + 1);
        let aligned = vary_over_missing_manual_axes(
            types.to_vec(),
            &types,
            &manual_variation_axes(&types),
            |mut input: ArrayType, axis, mesh| {
                if input.sharding().is_none() {
                    instruction_count.set(instruction_count.get() + 1);
                    let rank = input.rank();
                    input = input
                        .with_sharding(Sharding::replicated(mesh.clone(), rank))
                        .map_err(|error| TypeError::invalid(error.to_string()))?;
                }
                instruction_count.set(instruction_count.get() + 1);
                let operation = ParallelVaryOperation::new(axis.to_string());
                Ok(operation.infer_output_types(std::slice::from_ref(&input), &[])?.remove(0))
            },
        )?;
        self.instruction_count = instruction_count.get();
        let operation = DotOperation::new(dimensions.clone())
            .with_precision(self.precision)
            .with_accumulation_type(self.accumulation_type)
            .with_output_sharding(output_sharding.cloned());
        Ok(operation.infer_output_types(&aligned, &[])?.remove(0))
    }

    fn transpose(&mut self, value: &ArrayType, permutation: Vec<usize>) -> Result<ArrayType, ProgramError> {
        self.instruction_count += 1;
        value.transpose(permutation)
    }

    fn convert_element_type(&mut self, value: &ArrayType, data_type: DataType) -> Result<ArrayType, ProgramError> {
        self.instruction_count += 1;
        let operation = ConvertElementTypeOperation::<ArrayType>::new(data_type, false);
        Ok(operation.infer_output_types(std::slice::from_ref(value), &[])?.remove(0))
    }
}

/// State of one run of [`EinsumPlan::execute`], which holds the element type conversions that the steps share.
struct EinsumExecution<'s, S> {
    /// Primitive steps that the run computes, stages, or infers.
    steps: &'s mut S,

    /// Optional accumulation type that sums, mixed contractions, and the output are converted to.
    accumulation_type: Option<DataType>,
}

impl<S> EinsumExecution<'_, S> {
    /// Converts `value` to `data_type`, unless it already has that element type.
    fn convert<V>(&mut self, value: V, data_type: DataType) -> Result<V, ProgramError>
    where
        S: EinsumSteps<V>,
    {
        if self.steps.array_type(&value).data_type() == data_type {
            Ok(value)
        } else {
            self.steps.convert_element_type(&value, data_type)
        }
    }

    /// Sums `value` along `axes` after converting it to the accumulation type when one is requested, unless the
    /// element type of `value` is the type that the two promote to. This is the conversion rule of JAX, which sums at
    /// the element type of `value` only when the accumulation type adds nothing to it. Types without a common
    /// promotion (i.e., the 8-bit floating-point types with any other type) are converted as well.
    fn sum<V>(&mut self, value: V, axes: &[usize]) -> Result<V, ProgramError>
    where
        S: EinsumSteps<V>,
    {
        let data_type = self.steps.array_type(&value).data_type();
        let value = match self.accumulation_type {
            Some(accumulation_type)
                if data_type != accumulation_type
                    && DataType::promoted(&[data_type, accumulation_type]).ok() != Some(data_type) =>
            {
                self.steps.convert_element_type(&value, accumulation_type)?
            }
            _ => value,
        };
        self.steps.reduce_sum(&value, axes)
    }

    /// Sums the labels `uniques`, each of which names one axis of `value`, with a single reduction.
    fn sum_uniques<V>(
        &mut self,
        value: V,
        mut names: Vec<char>,
        uniques: Vec<char>,
    ) -> Result<(V, Vec<char>), ProgramError>
    where
        S: EinsumSteps<V>,
    {
        if uniques.is_empty() {
            return Ok((value, names));
        }
        let axes = uniques
            .iter()
            .map(|unique| names.iter().position(|name| name == unique).unwrap())
            .collect::<Vec<_>>();
        let value = self.sum(value, &axes)?;
        names.retain(|name| !uniques.contains(name));
        Ok((value, names))
    }

    /// Replaces every label that `counts` records as repeated in `value` by its diagonal, which takes the place of
    /// the last axis of the label, and sums the diagonals of the labels that `keep_names` does not contain.
    fn sum_repeats<V>(
        &mut self,
        mut value: V,
        mut names: Vec<char>,
        counts: &[(char, usize)],
        keep_names: &[char],
    ) -> Result<(V, Vec<char>), ProgramError>
    where
        S: EinsumSteps<V>,
    {
        for (repeated, _) in counts.iter().filter(|(_, count)| *count > 1) {
            loop {
                let axes = (0..names.len()).filter(|axis| names[*axis] == *repeated).collect::<Vec<_>>();
                let [removed_axis, .., kept_axis] = axes.as_slice() else { break };
                value = self.steps.diagonal(&value, *kept_axis, *removed_axis)?;
                names.remove(*removed_axis);
            }
            if !keep_names.contains(repeated) {
                let axis = names.iter().position(|name| name == repeated).unwrap();
                value = self.sum(value, &[axis])?;
                names.remove(axis);
            }
        }
        Ok((value, names))
    }

    /// Squeezes away every axis of `value` that has extent one while the other input of its step has another extent
    /// for its label, which broadcasts `value` along that label. Only static extents are compared.
    fn filter_singleton_axes<V>(
        &mut self,
        value: V,
        mut names: Vec<char>,
        other_type: &ArrayType,
        other_names: &[char],
    ) -> Result<(V, Vec<char>), ProgramError>
    where
        S: EinsumSteps<V>,
    {
        let value_type = self.steps.array_type(&value);
        let squeezed_axes = (0..names.len())
            .filter(|axis| {
                value_type.dimension(*axis) == Dimension::Static(1)
                    && other_names
                        .iter()
                        .position(|name| *name == names[*axis])
                        .is_some_and(|other_axis| other_type.dimension(other_axis) != Dimension::Static(1))
            })
            .collect::<Vec<_>>();
        if squeezed_axes.is_empty() {
            return Ok((value, names));
        }
        let value = self.steps.squeeze(&value, squeezed_axes.clone())?;
        for axis in squeezed_axes.into_iter().rev() {
            names.remove(axis);
        }
        Ok((value, names))
    }

    /// Contracts `lhs` and `rhs` after converting both to the accumulation type when one is requested and a `dot`
    /// cannot accumulate them at it itself. A `dot` accumulates two inputs of one element type as
    /// [`is_dot_accumulation_type`] describes, and two distinct 8-bit floating-point types natively. Inputs of
    /// different element types are converted otherwise, which happens when an earlier result at the accumulation type
    /// meets an input. Inputs of one element type are converted when JAX accepts the accumulation type for them
    /// (i.e., integers at a type that is at least as wide, and every type that promotes to it), and the remaining
    /// combinations are left for the `dot` to reject.
    fn contract<V: Clone>(
        &mut self,
        lhs: &V,
        rhs: &V,
        dimensions: &DotDimensionNumbers,
        output_sharding: Option<&Sharding>,
    ) -> Result<V, ProgramError>
    where
        S: EinsumSteps<V>,
    {
        let lhs_type = self.steps.array_type(lhs).data_type();
        let rhs_type = self.steps.array_type(rhs).data_type();
        let converts = self.accumulation_type.filter(|accumulation_type| {
            if lhs_type != rhs_type {
                !(is_native_f8_dot_type(lhs_type) && is_native_f8_dot_type(rhs_type))
            } else if is_dot_accumulation_type(lhs_type, *accumulation_type) {
                false
            } else if lhs_type.is_integer() {
                accumulation_type.bit_width() >= lhs_type.bit_width()
            } else {
                lhs_type.is_promotable_to(*accumulation_type)
            }
        });
        match converts {
            Some(accumulation_type) => {
                let lhs = self.convert(lhs.clone(), accumulation_type)?;
                let rhs = self.convert(rhs.clone(), accumulation_type)?;
                self.steps.contract(&lhs, &rhs, dimensions, output_sharding)
            }
            None => self.steps.contract(lhs, rhs, dimensions, output_sharding),
        }
    }
}

/// Model of a CPython `set` of small non-negative integers, which reproduces the order in which CPython iterates
/// such a set. The exhaustive path searches of `opt_einsum` enumerate the candidate pairs of the remaining inputs in
/// that order, and the order decides both which of several equally good paths they return and which candidates
/// their pruning discards. Reproducing it is therefore what makes [`EinsumPathStrategy::Auto`] select the paths that
/// JAX selects. The model follows `Objects/setobject.c` for the three operations that the searches use (i.e.,
/// construction from a sequence, difference, and union). A small integer is its own hash.
#[derive(Clone)]
struct PythonIntegerSet {
    /// Open addressing table, whose length is a power of two.
    table: Vec<PythonIntegerSetSlot>,

    /// Number of slots that are not empty.
    fill: usize,

    /// Number of slots that hold a value.
    used: usize,
}

/// One slot of the table of a [`PythonIntegerSet`].
#[derive(Copy, Clone, PartialEq, Eq)]
enum PythonIntegerSetSlot {
    /// Slot that never held a value.
    Empty,

    /// Slot whose value was removed.
    Dummy,

    /// Slot that holds a value.
    Active(usize),
}

impl PythonIntegerSet {
    /// Number of slots of an empty set.
    const MINIMUM_SIZE: usize = 8;

    /// Number of slots after the first probed one that are probed linearly.
    const LINEAR_PROBES: usize = 9;

    /// Number of bits by which the perturbation of the probe sequence is shifted at every step.
    const PERTURB_SHIFT: u32 = 5;

    /// Creates an empty set.
    fn new() -> Self {
        Self { table: vec![PythonIntegerSetSlot::Empty; Self::MINIMUM_SIZE], fill: 0, used: 0 }
    }

    /// Creates a set by adding `values` in order, like `set(values)` does for a sequence and like a set display does.
    fn from_values<I: IntoIterator<Item = usize>>(values: I) -> Self {
        let mut set = Self::new();
        for value in values {
            set.add(value);
        }
        set
    }

    /// Returns the number of values in this set.
    fn len(&self) -> usize {
        self.used
    }

    /// Returns the values of this set in the order in which CPython iterates them, which is the order of the table.
    fn values(&self) -> Vec<usize> {
        self.table
            .iter()
            .filter_map(|slot| match slot {
                PythonIntegerSetSlot::Active(value) => Some(*value),
                _ => None,
            })
            .collect()
    }

    /// Returns `true` if this set contains `value`.
    fn contains(&self, value: usize) -> bool {
        self.table.contains(&PythonIntegerSetSlot::Active(value))
    }

    /// Returns the difference of this set and `other`, like `self - other` does.
    fn difference(&self, other: &Self) -> Self {
        // A much larger set is copied and the values of `other` are discarded from the copy, which leaves dummies.
        // Otherwise the values that `other` does not contain are added to a new set in the order of this set.
        if (self.used >> 2) > other.used {
            let mut result = self.copy();
            for slot in &mut result.table {
                if matches!(slot, PythonIntegerSetSlot::Active(value) if other.contains(*value)) {
                    *slot = PythonIntegerSetSlot::Dummy;
                    result.used -= 1;
                }
            }
            if result.fill - result.used > (result.table.len() - 1) / 4 {
                result.resize(if result.used > 50000 { result.used * 2 } else { result.used * 4 });
            }
            result
        } else {
            let mut result = Self::new();
            for value in self.values() {
                if !other.contains(value) {
                    result.add(value);
                }
            }
            result
        }
    }

    /// Returns the union of this set and `other`, like `self | other` does.
    fn union(&self, other: &Self) -> Self {
        let mut result = self.copy();
        result.merge(other);
        result
    }

    /// Returns a copy of this set, like `set(self)` does.
    fn copy(&self) -> Self {
        let mut copy = Self::new();
        copy.merge(self);
        copy
    }

    /// Adds `value` to this set, like `set_add_entry` does.
    fn add(&mut self, value: usize) {
        let mask = self.table.len() - 1;
        let mut index = value & mask;
        let mut perturb = value;
        let mut free_slot = None;
        loop {
            let probes = if index + Self::LINEAR_PROBES <= mask { Self::LINEAR_PROBES } else { 0 };
            for slot in index..=index + probes {
                match self.table[slot] {
                    PythonIntegerSetSlot::Active(other) if other == value => return,
                    PythonIntegerSetSlot::Active(_) => {}
                    PythonIntegerSetSlot::Dummy => free_slot = Some(slot),
                    PythonIntegerSetSlot::Empty => {
                        self.used += 1;
                        if let Some(free_slot) = free_slot {
                            self.table[free_slot] = PythonIntegerSetSlot::Active(value);
                            return;
                        }
                        self.table[slot] = PythonIntegerSetSlot::Active(value);
                        self.fill += 1;
                        if self.fill * 5 >= mask * 3 {
                            self.resize(if self.used > 50000 { self.used * 2 } else { self.used * 4 });
                        }
                        return;
                    }
                }
            }
            perturb >>= Self::PERTURB_SHIFT;
            index = (index * 5 + 1 + perturb) & mask;
        }
    }

    /// Inserts `value`, which `table` does not contain, into `table`, which has no dummies, like `set_insert_clean`
    /// does.
    fn insert_clean(table: &mut [PythonIntegerSetSlot], value: usize) {
        let mask = table.len() - 1;
        let mut index = value & mask;
        let mut perturb = value;
        loop {
            let probes = if index + Self::LINEAR_PROBES <= mask { Self::LINEAR_PROBES } else { 0 };
            if let Some(slot) = (index..=index + probes).find(|slot| table[*slot] == PythonIntegerSetSlot::Empty) {
                table[slot] = PythonIntegerSetSlot::Active(value);
                return;
            }
            perturb >>= Self::PERTURB_SHIFT;
            index = (index * 5 + 1 + perturb) & mask;
        }
    }

    /// Rebuilds the table of this set with more than `minimum_used` slots, like `set_table_resize` does, inserting
    /// its values in the order of the old table and dropping its dummies.
    fn resize(&mut self, minimum_used: usize) {
        let mut size = Self::MINIMUM_SIZE;
        while size <= minimum_used {
            size <<= 1;
        }
        let mut table = vec![PythonIntegerSetSlot::Empty; size];
        for value in self.values() {
            Self::insert_clean(&mut table, value);
        }
        self.table = table;
        self.fill = self.used;
    }

    /// Adds the values of `other` to this set, like `set_merge` does.
    fn merge(&mut self, other: &Self) {
        if other.used == 0 {
            return;
        }
        if (self.fill + other.used) * 5 >= (self.table.len() - 1) * 3 {
            self.resize((self.used + other.used) * 2);
        }
        if self.fill == 0 && self.table.len() == other.table.len() && other.fill == other.used {
            // An empty set with a table of the same size and no dummies to drop copies the table as it is.
            self.table = other.table.clone();
            self.fill = other.fill;
            self.used = other.used;
        } else if self.fill == 0 {
            for value in other.values() {
                Self::insert_clean(&mut self.table, value);
            }
            self.fill = other.used;
            self.used = other.used;
        } else {
            for value in other.values() {
                self.add(value);
            }
        }
    }
}

/// Search state of [`optimal_ssa_path`] and [`branch_ssa_path`], which explore pairwise contraction orders depth
/// first. Both are ports of `opt_einsum.paths` and deliberately reproduce its caches, which key the result of a
/// candidate contraction by the labels of its two inputs alone, although that result also depends on the inputs that
/// remain when the candidate is first assessed.
struct PathSearch<'o> {
    /// Labels of the output of the einsum.
    output: &'o LabelSet,

    /// Extent of every label.
    sizes: &'o BTreeMap<char, u128>,

    /// Labels of every input and of every intermediate result on the current path, indexed by their identifiers.
    /// Identifiers are never reused (i.e., they are in static single assignment form).
    inputs: Vec<LabelSet>,

    /// Pairwise contractions of the current path.
    path: Vec<(usize, usize)>,

    /// Number of scalar operations and largest intermediate size of the best complete path found so far.
    best: Option<(u128, u128)>,

    /// Best complete path found so far.
    best_path: Vec<(usize, usize)>,

    /// Fewest scalar operations with which any explored path reached each depth, which the branching search uses to
    /// abandon paths that fall far behind.
    best_progress: HashMap<usize, u128>,

    /// Result labels and scalar operation count of each candidate contraction assessed so far.
    result_cache: HashMap<(LabelSet, LabelSet), (LabelSet, u128)>,
}

impl<'o> PathSearch<'o> {
    /// Creates the state of a search over `inputs`.
    fn new(inputs: &[LabelSet], output: &'o LabelSet, sizes: &'o BTreeMap<char, u128>) -> Self {
        Self {
            output,
            sizes,
            inputs: inputs.to_vec(),
            path: Vec::new(),
            best: None,
            best_path: Vec::new(),
            best_progress: HashMap::new(),
            result_cache: HashMap::new(),
        }
    }

    /// Returns the result labels and the scalar operation count of contracting the inputs `first` and `second` while
    /// `remaining` inputs are left, like `calc_k12_flops` does, caching both by the labels of the two inputs.
    fn assess(&mut self, remaining: &PythonIntegerSet, first: usize, second: usize) -> (LabelSet, u128) {
        let key = (self.inputs[first].clone(), self.inputs[second].clone());
        if let Some(result) = self.result_cache.get(&key) {
            return result.clone();
        }
        let either = key.0.union(&key.1).copied().collect::<LabelSet>();
        let keep = remaining
            .values()
            .into_iter()
            .filter(|input| *input != first && *input != second)
            .flat_map(|input| &self.inputs[input])
            .chain(self.output)
            .copied()
            .collect::<LabelSet>();
        let has_inner_product = key.0.intersection(&key.1).any(|label| !keep.contains(label));
        let flops = size_of(&either, self.sizes).saturating_mul(if has_inner_product { 2 } else { 1 });
        let result = (either.intersection(&keep).copied().collect::<LabelSet>(), flops);
        self.result_cache.insert(key, result.clone());
        result
    }

    /// Returns the inputs that remain after contracting `first` and `second` into a new intermediate result, like
    /// `remaining - {first, second} | {len(inputs)}` does.
    fn contracted(&self, remaining: &PythonIntegerSet, first: usize, second: usize) -> PythonIntegerSet {
        remaining
            .difference(&PythonIntegerSet::from_values([first, second]))
            .union(&PythonIntegerSet::from_values([self.inputs.len()]))
    }

    /// Returns the pairs of the `remaining` inputs in the order in which `itertools.combinations` enumerates them,
    /// each with its smaller identifier first.
    fn candidates(remaining: &PythonIntegerSet) -> Vec<(usize, usize)> {
        let remaining = remaining.values();
        (0..remaining.len())
            .flat_map(|first| (first + 1..remaining.len()).map(move |second| (first, second)))
            .map(|(first, second)| (remaining[first].min(remaining[second]), remaining[first].max(remaining[second])))
            .collect()
    }

    /// Explores every pairwise contraction of the `remaining` inputs, keeping the first complete path with the fewest
    /// scalar operations, like `_optimal_iterate` does.
    fn iterate_optimal(&mut self, remaining: &PythonIntegerSet, flops: u128) {
        if remaining.len() == 1 {
            self.best = Some((flops, 0));
            self.best_path = self.path.clone();
            return;
        }
        for (first, second) in Self::candidates(remaining) {
            let (result, result_flops) = self.assess(remaining, first, second);
            let flops = flops.saturating_add(result_flops);
            if self.best.is_some_and(|(best_flops, _)| flops >= best_flops) {
                continue;
            }
            let contracted = self.contracted(remaining, first, second);
            self.path.push((first, second));
            self.inputs.push(result);
            self.iterate_optimal(&contracted, flops);
            self.inputs.pop();
            self.path.pop();
        }
    }

    /// Explores the `branch_count` most promising pairwise contractions of the `remaining` inputs, or all of them
    /// without a `branch_count`, like `BranchBound._branch_iterate` does with its default options.
    fn iterate_branch(&mut self, remaining: &PythonIntegerSet, flops: u128, size: u128, branch_count: Option<usize>) {
        if remaining.len() == 1 {
            self.best = Some((flops, size));
            self.best_path = self.path.clone();
            return;
        }

        // Candidates are ordered by the memory that they free, then by their own scalar operations, and then by the
        // totals of the path that they extend. Outer products are only considered when nothing else is left.
        let depth = self.inputs.len();
        let mut candidates = Vec::new();
        for only_connected in [true, false] {
            for (first, second) in Self::candidates(remaining) {
                if only_connected && self.inputs[first].is_disjoint(&self.inputs[second]) {
                    continue;
                }
                let (result, result_flops) = self.assess(remaining, first, second);
                let result_size = size_of(&result, self.sizes);
                let new_flops = flops.saturating_add(result_flops);
                let new_size = size.max(result_size);
                if self.best.is_some_and(|best| (new_flops, new_size) >= best) {
                    continue;
                }

                // A path is abandoned once it needs more than four times the scalar operations of the best path
                // that reached this depth so far, which depends on the order in which candidates are assessed.
                match self.best_progress.get(&depth) {
                    Some(progress) if new_flops >= *progress => {
                        if new_flops > progress.saturating_mul(4) {
                            continue;
                        }
                    }
                    _ => {
                        self.best_progress.insert(depth, new_flops);
                    }
                }
                let first_size = size_of(&self.inputs[first], self.sizes);
                let second_size = size_of(&self.inputs[second], self.sizes);
                let cost = signed(result_size) - signed(first_size) - signed(second_size);
                candidates.push((cost, result_flops, new_flops, new_size, first, second, result));
            }
            if !candidates.is_empty() {
                break;
            }
        }
        candidates.sort_by(|left, right| {
            (left.0, left.1, left.2, left.3, left.4, left.5)
                .cmp(&(right.0, right.1, right.2, right.3, right.4, right.5))
        });
        for (_, _, new_flops, new_size, first, second, result) in
            candidates.into_iter().take(branch_count.unwrap_or(usize::MAX))
        {
            let contracted = self.contracted(remaining, first, second);
            self.path.push((first, second));
            self.inputs.push(result);
            self.iterate_branch(&contracted, new_flops, new_size, branch_count);
            self.inputs.pop();
            self.path.pop();
        }
    }
}

/// Returns the pairwise contractions with the fewest scalar operations for the `inputs` and `output` of an einsum, in
/// static single assignment form, by exploring every contraction order. This is a port of `opt_einsum.paths.optimal`
/// without a memory limit.
fn optimal_ssa_path(inputs: &[LabelSet], output: &LabelSet, sizes: &BTreeMap<char, u128>) -> Vec<(usize, usize)> {
    let mut search = PathSearch::new(inputs, output, sizes);
    search.iterate_optimal(&PythonIntegerSet::from_values(0..inputs.len()), 0);
    search.best_path
}

/// Returns pairwise contractions for the `inputs` and `output` of an einsum, in static single assignment form, by
/// exploring the `branch_count` most promising contractions at every step, or all of them without a `branch_count`.
/// This is a port of `opt_einsum.paths.branch` with its default options and without a memory limit.
fn branch_ssa_path(
    inputs: &[LabelSet],
    output: &LabelSet,
    sizes: &BTreeMap<char, u128>,
    branch_count: Option<usize>,
) -> Vec<(usize, usize)> {
    let mut search = PathSearch::new(inputs, output, sizes);
    search.iterate_branch(&PythonIntegerSet::from_values(0..inputs.len()), 0, 0, branch_count);
    search.best_path
}

/// Returns pairwise contractions for the `inputs` and `output` of an einsum, in static single assignment form, by
/// first multiplying inputs with equal labels, then repeatedly contracting the pair of inputs that frees the most
/// memory, and finally taking outer products from the smallest results up. This is a port of
/// `opt_einsum.paths.ssa_greedy_optimize` with its default options.
fn greedy_ssa_path(inputs: &[LabelSet], output: &LabelSet, sizes: &BTreeMap<char, u128>) -> Vec<(usize, usize)> {
    /// Candidate contraction of two remaining inputs, ordered by its cost and then by the identifiers of its inputs,
    /// which makes the order of the candidates total.
    type Candidate = (i128, usize, usize, LabelSet, LabelSet, LabelSet);

    // A label that every input contains can only be contracted in the last step, so it is treated like an output
    // label, which avoids comparing all pairs of the inputs that share it.
    let common = inputs.iter().skip(1).fold(inputs[0].clone(), |common, input| &common & input);
    let output = output | &common;

    // Inputs with equal labels are multiplied elementwise first.
    let mut next_identifier = inputs.len();
    let mut path = Vec::new();
    let mut remaining = BTreeMap::<LabelSet, usize>::new();
    for (identifier, input) in inputs.iter().enumerate() {
        if let Some(existing) = remaining.get_mut(input) {
            path.push((*existing, identifier));
            *existing = next_identifier;
            next_identifier += 1;
        } else {
            remaining.insert(input.clone(), identifier);
        }
    }

    // The inputs that contain each label that can be contracted, and the labels that at least two and at least three
    // inputs contain, which decide whether a contraction can remove a label.
    let mut label_inputs = BTreeMap::<char, BTreeSet<LabelSet>>::new();
    for input in remaining.keys() {
        for label in input.difference(&output) {
            label_inputs.entry(*label).or_default().insert(input.clone());
        }
    }
    let mut shared_by_two = label_inputs.iter().filter(|(_, inputs)| inputs.len() >= 2).map(|(label, _)| *label);
    let mut shared_by_two = shared_by_two.by_ref().collect::<LabelSet>();
    let mut shared_by_three = label_inputs
        .iter()
        .filter(|(_, inputs)| inputs.len() >= 3)
        .map(|(label, _)| *label)
        .collect::<LabelSet>();
    let mut footprints =
        remaining.keys().map(|input| (input.clone(), size_of(input, sizes))).collect::<HashMap<_, _>>();

    // The best candidate of an input among `others` is the one that frees the most memory, where ties are broken by
    // the identifiers of its inputs.
    let best_candidate = |first: &LabelSet,
                          others: &[LabelSet],
                          remaining: &BTreeMap<LabelSet, usize>,
                          footprints: &HashMap<LabelSet, u128>,
                          shared_by_two: &LabelSet,
                          shared_by_three: &LabelSet|
     -> Candidate {
        others
            .iter()
            .map(|second| {
                let either = first | second;
                let both = first & second;
                let result = either
                    .iter()
                    .filter(|label| {
                        output.contains(label)
                            || (both.contains(label) && shared_by_three.contains(label))
                            || (!both.contains(label) && shared_by_two.contains(label))
                    })
                    .copied()
                    .collect::<LabelSet>();
                let cost = signed(size_of(&result, sizes)) - signed(footprints[first]) - signed(footprints[second]);
                let (first, second) =
                    if remaining[first] > remaining[second] { (second, first) } else { (first, second) };
                (cost, remaining[second], remaining[first], first.clone(), second.clone(), result)
            })
            .min()
            .unwrap()
    };

    let mut queue = BinaryHeap::new();
    for inputs in label_inputs.values() {
        let mut inputs = inputs.iter().cloned().collect::<Vec<_>>();
        inputs.sort_by_key(|input| remaining[input]);
        for index in 0..inputs.len().saturating_sub(1) {
            queue.push(Reverse(best_candidate(
                &inputs[index],
                &inputs[index + 1..],
                &remaining,
                &footprints,
                &shared_by_two,
                &shared_by_three,
            )));
        }
    }

    while let Some(Reverse((_, _, _, first, second, result))) = queue.pop() {
        // A candidate is obsolete once one of its inputs was contracted.
        if !remaining.contains_key(&first) || !remaining.contains_key(&second) {
            continue;
        }
        let first_identifier = remaining.remove(&first).unwrap();
        let second_identifier = remaining.remove(&second).unwrap();
        for input in [&first, &second] {
            for label in input.difference(&output) {
                label_inputs.get_mut(label).unwrap().remove(input);
            }
        }
        path.push((first_identifier, second_identifier));
        if let Some(existing) = remaining.get(&result) {
            path.push((*existing, next_identifier));
            next_identifier += 1;
        } else {
            for label in result.difference(&output) {
                label_inputs.entry(*label).or_default().insert(result.clone());
            }
        }
        remaining.insert(result.clone(), next_identifier);
        next_identifier += 1;
        for label in first.iter().chain(second.difference(&output)) {
            let count = label_inputs.get(label).map_or(0, BTreeSet::len);
            if count >= 2 {
                shared_by_two.insert(*label);
            } else {
                shared_by_two.remove(label);
            }
            if count >= 3 {
                shared_by_three.insert(*label);
            } else {
                shared_by_three.remove(label);
            }
        }
        footprints.insert(result.clone(), size_of(&result, sizes));

        // The new result is a candidate with every remaining input that shares a label with it.
        let others = result
            .iter()
            .filter_map(|label| label_inputs.get(label))
            .flatten()
            .filter(|other| **other != result)
            .cloned()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        if !others.is_empty() {
            queue.push(Reverse(best_candidate(
                &result,
                &others,
                &remaining,
                &footprints,
                &shared_by_two,
                &shared_by_three,
            )));
        }
    }

    // The remaining results share no label that can be contracted, so their outer products are taken from the
    // smallest results up.
    let mut queue = remaining
        .iter()
        .map(|(input, identifier)| Reverse((size_of(&(input & &output), sizes), *identifier, input.clone())))
        .collect::<BinaryHeap<_>>();
    let Reverse((_, mut first_identifier, mut first)) = queue.pop().unwrap();
    while let Some(Reverse((_, second_identifier, second))) = queue.pop() {
        path.push((first_identifier.min(second_identifier), first_identifier.max(second_identifier)));
        let result = &(&first | &second) & &output;
        queue.push(Reverse((size_of(&result, sizes), next_identifier, result)));
        next_identifier += 1;
        Reverse((_, first_identifier, first)) = queue.pop().unwrap();
    }
    path
}

/// Converts pairwise contractions in static single assignment form, whose identifiers are never reused, into the
/// positions that [`EinsumPathStrategy::Explicit`] uses, which refer to the list of the remaining inputs. This is a
/// port of `opt_einsum.paths.ssa_to_linear`.
fn ssa_to_linear(ssa_path: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut identifiers = (0..=ssa_path.len()).collect::<Vec<_>>();
    let mut next_identifier = ssa_path.len() + 1;
    let mut path = Vec::with_capacity(ssa_path.len());
    for (first, second) in ssa_path {
        let first = identifiers.partition_point(|identifier| identifier < first);
        let second = identifiers.partition_point(|identifier| identifier < second);
        let (first, second) = (first.min(second), first.max(second));
        identifiers.remove(second);
        identifiers.remove(first);
        identifiers.push(next_identifier);
        next_identifier += 1;
        path.push((first, second));
    }
    path
}

/// Returns the number of elements of an array whose axes have the provided labels, saturating on overflow.
fn size_of(labels: &LabelSet, sizes: &BTreeMap<char, u128>) -> u128 {
    labels.iter().fold(1u128, |size, label| size.saturating_mul(sizes[label]))
}

/// Converts a size to a signed integer, saturating on overflow.
fn signed(size: u128) -> i128 {
    i128::try_from(size).unwrap_or(i128::MAX)
}

/// Returns how often each label appears in `names`, in the order of their first appearances.
fn count_names(names: &[char]) -> Vec<(char, usize)> {
    let mut counts = Vec::<(char, usize)>::new();
    for name in names {
        match counts.iter_mut().find(|(other, _)| other == name) {
            Some((_, count)) => *count += 1,
            None => counts.push((*name, 1)),
        }
    }
    counts
}

/// Returns how often `name` appears according to `counts`.
fn count_of(counts: &[(char, usize)], name: &char) -> usize {
    counts.iter().find(|(other, _)| other == name).map_or(0, |(_, count)| *count)
}

/// Returns the symbol with the provided index in the sequence of label symbols of `opt_einsum` (i.e., `a` through
/// `z`, then `A` through `Z`, and then further Unicode characters).
fn einsum_symbol(index: usize) -> char {
    const BASE: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    match index {
        ..52 => BASE[index] as char,
        // The surrogate code points are skipped, because they are not characters.
        55296.. => char::from_u32(index as u32 + 2048).unwrap_or(char::REPLACEMENT_CHARACTER),
        _ => char::from_u32(index as u32 + 140).unwrap_or(char::REPLACEMENT_CHARACTER),
    }
}

/// Returns the [`CompositeKey`] of the named composite of an einsum with the provided specification and options.
/// The attributes encode everything that determines its computation for given input types. Labels are encoded by
/// their code points and an ellipsis by `-1`, the path strategy by its position among the strategies followed by the
/// positions of an explicit path, and the precision by its kind followed by its fields.
fn einsum_composite_key(specification: &EinsumSpecification, options: &EinsumOptions) -> CompositeKey {
    let integer = |value: usize| CompositeAttribute::Integer(value as i128);
    let subscript = |subscript: &[EinsumLabel]| {
        CompositeAttribute::Sequence(
            subscript
                .iter()
                .map(|label| match label {
                    EinsumLabel::Named(label) => CompositeAttribute::Integer(u32::from(*label) as i128),
                    EinsumLabel::Ellipsis => CompositeAttribute::Integer(-1),
                })
                .collect(),
        )
    };
    let path = match &options.path {
        EinsumPathStrategy::Auto => vec![integer(0)],
        EinsumPathStrategy::Optimal => vec![integer(1)],
        EinsumPathStrategy::Greedy => vec![integer(2)],
        EinsumPathStrategy::Unoptimized => vec![integer(3)],
        EinsumPathStrategy::Explicit(path) => std::iter::once(integer(4))
            .chain(path.iter().flat_map(|(first, second)| [integer(*first), integer(*second)]))
            .collect(),
    };
    let precision = match options.precision {
        DotPrecision::Inputs { lhs, rhs } => vec![integer(0), integer(lhs as usize), integer(rhs as usize)],
        DotPrecision::Algorithm(algorithm) => vec![
            integer(1),
            CompositeAttribute::DataType(algorithm.lhs_precision_type()),
            CompositeAttribute::DataType(algorithm.rhs_precision_type()),
            CompositeAttribute::DataType(algorithm.accumulation_type()),
            integer(algorithm.lhs_component_count()),
            integer(algorithm.rhs_component_count()),
            integer(algorithm.primitive_operation_count()),
            CompositeAttribute::Boolean(algorithm.allow_imprecise_accumulation()),
        ],
        DotPrecision::Preset(preset) => vec![integer(2), integer(preset as usize)],
    };
    CompositeKey::new(EINSUM_COMPOSITE_NAME, EINSUM_COMPOSITE_VERSION)
        .with_attribute(
            "inputs",
            CompositeAttribute::Sequence(specification.inputs.iter().map(|input| subscript(input)).collect()),
        )
        .with_attribute(
            "output",
            CompositeAttribute::Optional(specification.output.as_deref().map(|output| Box::new(subscript(output)))),
        )
        .with_attribute("path", CompositeAttribute::Sequence(path))
        .with_attribute("accumulation_type", options.accumulation_type)
        .with_attribute("precision", CompositeAttribute::Sequence(precision))
}

#[cfg(test)]
mod tests {
    use half::{bf16, f16};
    use indoc::{formatdoc, indoc};
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrValue, ArrayOperation, DimensionBounds, DimensionValue, DimensionVariable,
        LogicalMesh, MeshAxis, MeshAxisType, Shape, ShardingDimension,
    };
    use crate::axes::NamedAxis;
    use crate::batching::{BatchAxis, batch};
    use crate::contexts::EagerContext;
    use crate::differentiation::differentiate_at;
    use crate::macros::check_gradient;
    use crate::operations::arithmetic::Add;
    use crate::operations::dot::{DotAlgorithm, DotAlgorithmPreset};
    use crate::partial::{PartialEvaluationOutput, PartialValue};
    use crate::programs::Provenance;
    use crate::tracing::{DomainTracer, DomainTracingContext, Trace};

    use super::*;

    type ArrayContext = EagerContext<Array, ArrayOperation<Array>>;
    type ArrayIrEagerContext = EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

    /// Creates the input at position `index` of the reference einsums, which holds `((v * (index + 2)) % 7) - 3` for
    /// its row-major positions `v`, so that every reference result is an exactly representable integer.
    fn reference_input(index: usize, shape: &[usize]) -> Array {
        let count = shape.iter().product::<usize>();
        let values = (0..count).map(|value| ((value * (index + 2)) % 7) as f64 - 3.0).collect::<Vec<_>>();
        Array::from_elements(ArrayType::new_static(DataType::F64, shape.to_vec()), &values).unwrap()
    }

    /// Plans `subscripts` for `f32` inputs with the provided static shapes and returns the selected path.
    fn planned_path(subscripts: &str, shapes: &[&[usize]], strategy: EinsumPathStrategy) -> Vec<(usize, usize)> {
        let input_types =
            shapes.iter().map(|shape| ArrayType::new_static(DataType::F32, shape.to_vec())).collect::<Vec<_>>();
        let specification = subscripts.parse::<EinsumSpecification>().unwrap();
        specification.plan(&input_types, &EinsumOptions::new().with_path(strategy)).unwrap().path().to_vec()
    }

    #[test]
    fn test_einsum_label_from_index() {
        assert_eq!(EinsumLabel::from_index(0), Some(EinsumLabel::Named('a')));
        assert_eq!(EinsumLabel::from_index(25), Some(EinsumLabel::Named('z')));
        assert_eq!(EinsumLabel::from_index(26), Some(EinsumLabel::Named('A')));
        assert_eq!(EinsumLabel::from_index(51), Some(EinsumLabel::Named('Z')));
        assert_eq!(EinsumLabel::from_index(52), Some(EinsumLabel::Named('À')));

        // Characters that are not alphabetic are skipped (e.g., `×` between `Ö` and `Ø`), so that every label is
        // accepted by a specification, and the labels end with the alphabetic characters.
        assert_eq!(EinsumLabel::from_index(74), Some(EinsumLabel::Named('Ö')));
        assert_eq!(EinsumLabel::from_index(75), Some(EinsumLabel::Named('Ø')));
        let labels = (0..200).flat_map(EinsumLabel::from_index).collect::<Vec<_>>();
        assert_eq!(labels.len(), 200);
        assert!(EinsumSpecification::from_labels(vec![labels.clone()], Some(labels)).is_ok());
        assert_eq!(EinsumLabel::from_index(usize::MAX), None);
        assert_eq!(EinsumLabel::from('q'), EinsumLabel::Named('q'));
        assert_eq!(EinsumLabel::Named('q').to_string(), "q");
        assert_eq!(EinsumLabel::Ellipsis.to_string(), "...");
        assert_eq!(format!("{:?}", EinsumLabel::Named('q')), "Named('q')");
    }

    #[test]
    fn test_einsum_specification_from_labels() {
        let labels = |text: &str| text.chars().map(EinsumLabel::from).collect::<Vec<_>>();
        let specification =
            EinsumSpecification::from_labels(vec![labels("ij"), labels("jk")], Some(labels("ik"))).unwrap();
        assert_eq!(specification.inputs(), &[labels("ij"), labels("jk")]);
        assert_eq!(specification.output(), Some(labels("ik").as_slice()));
        assert_eq!(specification.to_string(), "ij,jk->ik");

        // The implicit form has no output, and integer label lists map to letters.
        let indices = |indices: &[usize]| indices.iter().flat_map(|index| EinsumLabel::from_index(*index)).collect();
        let implicit = EinsumSpecification::from_labels(vec![indices(&[0, 1]), indices(&[1, 2])], None).unwrap();
        assert_eq!(implicit.output(), None);
        assert_eq!(implicit.to_string(), "ab,bc");
        let ellipsis = vec![EinsumLabel::Ellipsis, EinsumLabel::Named('i')];
        let batched = EinsumSpecification::from_labels(vec![ellipsis.clone()], Some(vec![EinsumLabel::Ellipsis]));
        assert_eq!(batched.unwrap().to_string(), "...i->...");

        // Invalid label lists are rejected with the offending label.
        assert_eq!(
            EinsumSpecification::from_labels(Vec::new(), None),
            Err(TypeError::invalid("`einsum` subscripts must have at least one input")),
        );
        assert_eq!(
            EinsumSpecification::from_labels(vec![labels("i1")], None),
            Err(TypeError::invalid("`einsum` subscripts contain the invalid label `1`")),
        );
        assert_eq!(
            EinsumSpecification::from_labels(vec![vec![EinsumLabel::Ellipsis, EinsumLabel::Ellipsis]], None),
            Err(TypeError::invalid("`einsum` subscripts contain more than one ellipsis in one subscript")),
        );
        assert_eq!(
            EinsumSpecification::from_labels(vec![labels("ij")], Some(labels("ii"))),
            Err(TypeError::invalid("`einsum` output label `i` appears more than once")),
        );
        assert_eq!(
            EinsumSpecification::from_labels(vec![labels("ij")], Some(labels("ik"))),
            Err(TypeError::invalid("`einsum` output label `k` does not appear in any input")),
        );
    }

    #[test]
    fn test_einsum_specification_plan() {
        let types = |shapes: &[&[usize]]| {
            shapes.iter().map(|shape| ArrayType::new_static(DataType::F32, shape.to_vec())).collect::<Vec<_>>()
        };
        let plan = |subscripts: &str, shapes: &[&[usize]]| {
            subscripts.parse::<EinsumSpecification>().unwrap().plan(&types(shapes), &EinsumOptions::new())
        };
        let labels = |plan: &EinsumPlan| {
            let inputs = plan.input_labels().iter().map(|labels| labels.iter().collect()).collect::<Vec<String>>();
            (inputs.join(","), plan.output_labels().iter().collect::<String>())
        };

        // One input needs no pairwise contraction and two inputs need one.
        let trace = plan("ii", &[&[3, 3]]).unwrap();
        assert_eq!(labels(&trace), ("ii".to_string(), String::new()));
        assert_eq!(trace.path(), &[]);
        assert_eq!(trace.specification().to_string(), "ii");
        assert_eq!(trace.options(), &EinsumOptions::new());
        let matmul = plan("ij,jk->ik", &[&[2, 3], &[3, 4]]).unwrap();
        assert_eq!(labels(&matmul), ("ij,jk".to_string(), "ik".to_string()));
        assert_eq!(matmul.path(), &[(0, 1)]);

        // The implicit output holds the labels that appear once, in the order of their characters, so uppercase
        // labels precede lowercase ones.
        assert_eq!(labels(&plan("ba,Ab", &[&[2, 3], &[4, 2]]).unwrap()), ("ba,Ab".to_string(), "Aa".to_string()));
        assert_eq!(labels(&plan("ij,ji", &[&[2, 3], &[3, 2]]).unwrap()), ("ij,ji".to_string(), String::new()));

        // Ellipses expand to the last of as many unused symbols as the largest input rank, aligned at their last
        // axes. The implicit output starts with the axes of the longest ellipsis, even where an input repeats them.
        let batched = plan("...ij,...jk->...ik", &[&[5, 6, 2, 3], &[6, 3, 4]]).unwrap();
        assert_eq!(labels(&batched), ("cdij,djk".to_string(), "cdik".to_string()));
        let implicit = plan("...a,...a", &[&[2, 3, 4], &[3, 4]]).unwrap();
        assert_eq!(labels(&implicit), ("cda,da".to_string(), "cd".to_string()));
        assert_eq!(labels(&plan("...", &[&[2, 3]]).unwrap()), ("ab".to_string(), "ab".to_string()));
        assert_eq!(labels(&plan("i...->...", &[&[2]]).unwrap()), ("i".to_string(), String::new()));

        // A chain of matrix products contracts its smallest product first.
        assert_eq!(plan("ij,jk,kl->il", &[&[2, 3], &[3, 4], &[4, 2]]).unwrap().path(), &[(1, 2), (0, 1)]);

        // The number of inputs, their ranks, and their element types must match the subscripts.
        assert_eq!(
            plan("ij,jk", &[&[2, 3]]),
            Err(ProgramError::Type(TypeError::invalid("`einsum` subscripts `ij,jk` have 2 inputs but got 1"))),
        );
        assert_eq!(
            plan("ij,jk", &[&[2, 3], &[3]]),
            Err(ProgramError::Type(TypeError::invalid(
                "`einsum` subscript 1 of `ij,jk` does not match the rank 1 of its input",
            ))),
        );
        assert_eq!(
            plan("...ij", &[&[3]]),
            Err(ProgramError::Type(TypeError::invalid(
                "`einsum` subscript 0 of `...ij` does not match the rank 1 of its input",
            ))),
        );
        let specification = "ij,jk".parse::<EinsumSpecification>().unwrap();
        assert_eq!(
            specification.plan(
                &[ArrayType::new_static(DataType::F32, [2, 3]), ArrayType::new_static(DataType::F64, [3, 4])],
                &EinsumOptions::new(),
            ),
            Err(ProgramError::Type(TypeError::invalid(
                "`einsum` inputs must have the same element type but got `f32` and `f64`",
            ))),
        );

        // Every label has one extent, except for inputs of extent one, which are broadcast.
        assert_eq!(plan("ij,jk", &[&[2, 1], &[3, 4]]).unwrap().path(), &[(0, 1)]);
        assert_eq!(
            plan("ij,jk", &[&[2, 3], &[4, 5]]),
            Err(ProgramError::Type(TypeError::invalid(
                "`einsum` label `j` has extent `4` in input 1 but extent `3` in an earlier input",
            ))),
        );
        assert_eq!(
            plan("ii", &[&[1, 3]]),
            Err(ProgramError::Type(TypeError::invalid(
                "`einsum` label `i` is repeated in input 0 of type `f32[1, 3]` along axes with different extents",
            ))),
        );

        // Dynamic extents take part in the path search as extent 8, and labels must share the same dynamic extent.
        let extent = |name: &str| Dimension::Dynamic(DimensionVariable::new(name, DimensionBounds::unbounded()));
        let dynamic = |dimensions: Vec<Dimension>| ArrayType::new(DataType::F32, Shape::new(dimensions));
        let chain = "ij,jk,kl->il".parse::<EinsumSpecification>().unwrap();
        let n = extent("n");
        let dynamic_types =
            [dynamic(vec![2.into(), n.clone()]), dynamic(vec![n.clone(), 2.into()]), dynamic(vec![2.into(), 9.into()])];
        assert_eq!(
            chain.plan(&dynamic_types, &EinsumOptions::new()).unwrap().path(),
            chain.plan(&types(&[&[2, 8], &[8, 2], &[2, 9]]), &EinsumOptions::new()).unwrap().path(),
        );
        assert_eq!(
            chain.plan(
                &[dynamic(vec![2.into(), n]), dynamic(vec![extent("m"), 2.into()]), dynamic_types[2].clone()],
                &EinsumOptions::new(),
            ),
            Err(ProgramError::Type(TypeError::invalid(
                "`einsum` label `j` has extent `m` in input 1 but extent `n` in an earlier input",
            ))),
        );
    }

    #[test]
    fn test_einsum_specification_display() {
        for subscripts in ["ij,jk->ik", "ij,jk", "...ij,...jk->...ik", "i...j->j...i", ",i->i", "ii->"] {
            assert_eq!(subscripts.parse::<EinsumSpecification>().unwrap().to_string(), subscripts);
        }
        assert_eq!(
            format!("{:?}", "i->".parse::<EinsumSpecification>().unwrap()),
            "EinsumSpecification { inputs: [[Named('i')]], output: Some([]) }",
        );
    }

    #[test]
    fn test_einsum_specification_from_str() {
        let labels = |text: &str| text.chars().map(EinsumLabel::from).collect::<Vec<_>>();
        let specification = " ij , jk -> ik ".parse::<EinsumSpecification>().unwrap();
        assert_eq!(specification.inputs(), &[labels("ij"), labels("jk")]);
        assert_eq!(specification.output(), Some(labels("ik").as_slice()));

        // An ellipsis may appear anywhere in a subscript, an empty subscript denotes a rank-0 input or output, and
        // the implicit form has no output.
        let specification = "i...j,->...".parse::<EinsumSpecification>().unwrap();
        let ellipsis = vec![EinsumLabel::Named('i'), EinsumLabel::Ellipsis, EinsumLabel::Named('j')];
        assert_eq!(specification.inputs(), &[ellipsis, Vec::new()]);
        assert_eq!(specification.output(), Some([EinsumLabel::Ellipsis].as_slice()));
        assert_eq!("ij,jk".parse::<EinsumSpecification>().unwrap().output(), None);
        assert_eq!("ij->".parse::<EinsumSpecification>().unwrap().output(), Some([].as_slice()));

        // Malformed subscripts are rejected.
        let parse = |subscripts: &str| subscripts.parse::<EinsumSpecification>();
        let invalid_ellipsis = "`einsum` subscripts contain an invalid ellipsis in `i..j`";
        assert_eq!(parse("i..j"), Err(TypeError::invalid(invalid_ellipsis)));
        assert_eq!(parse("i."), Err(TypeError::invalid("`einsum` subscripts contain an invalid ellipsis in `i.`")));
        assert_eq!(
            parse("...i...->i"),
            Err(TypeError::invalid("`einsum` subscripts contain more than one ellipsis in one subscript")),
        );
        for subscripts in ["ij->ik->k", "ij-jk", "ij>jk", "i->j,k"] {
            assert_eq!(
                parse(subscripts),
                Err(TypeError::invalid("`einsum` subscripts must contain at most one `->`, after all of the inputs")),
            );
        }
        assert_eq!(parse("i2,2j"), Err(TypeError::invalid("`einsum` subscripts contain the invalid label `2`")));
        assert_eq!(parse("ij->jj"), Err(TypeError::invalid("`einsum` output label `j` appears more than once")));
        assert_eq!(parse("ij->k"), Err(TypeError::invalid("`einsum` output label `k` does not appear in any input")),);
    }

    #[test]
    fn test_einsum_path_strategy() {
        // The default strategy is the automatic one.
        assert_eq!(EinsumPathStrategy::default(), EinsumPathStrategy::Auto);
        assert_eq!(EinsumOptions::new().path(), &EinsumPathStrategy::Auto);

        // The unoptimized strategy contracts from left to right, and an explicit path is used as it is.
        let shapes: &[&[usize]] = &[&[2, 3], &[3, 4], &[4, 5], &[5, 2]];
        assert_eq!(
            planned_path("ij,jk,kl,lm->im", shapes, EinsumPathStrategy::Unoptimized),
            vec![(0, 1), (0, 1), (0, 1)],
        );
        assert_eq!(
            planned_path("ij,jk,kl,lm->im", shapes, EinsumPathStrategy::Explicit(vec![(2, 3), (0, 1), (0, 1)])),
            vec![(2, 3), (0, 1), (0, 1)],
        );
        assert_eq!(planned_path("ij,jk,kl,lm->im", shapes, EinsumPathStrategy::Optimal), vec![(2, 3), (1, 2), (0, 1)]);
        assert_eq!(planned_path("ij,jk,kl,lm->im", shapes, EinsumPathStrategy::Greedy), vec![(2, 3), (1, 2), (0, 1)]);

        // An explicit path must contract every input once with valid positions.
        let plan = |path: Vec<(usize, usize)>| {
            let types = shapes.iter().map(|shape| ArrayType::new_static(DataType::F32, shape.to_vec()));
            "ij,jk,kl,lm->im"
                .parse::<EinsumSpecification>()
                .unwrap()
                .plan(&types.collect::<Vec<_>>(), &EinsumOptions::new().with_path(EinsumPathStrategy::Explicit(path)))
        };
        assert_eq!(
            plan(vec![(0, 1), (0, 1)]).map(|_| ()),
            Err(ProgramError::Type(TypeError::invalid(
                "`einsum` path must have 3 pairwise contractions for 4 inputs but has 2",
            ))),
        );
        assert_eq!(
            plan(vec![(0, 1), (0, 3), (0, 1)]).map(|_| ()),
            Err(ProgramError::Type(TypeError::invalid(
                "`einsum` path contracts the positions 0 and 3, which must be two distinct positions among its 3 \
                 remaining inputs",
            ))),
        );
        assert_eq!(
            plan(vec![(1, 1), (0, 1), (0, 1)]).map(|_| ()),
            Err(ProgramError::Type(TypeError::invalid(
                "`einsum` path contracts the positions 1 and 1, which must be two distinct positions among its 4 \
                 remaining inputs",
            ))),
        );

        // One input takes an empty explicit path and two inputs take their one contraction in either order.
        assert_eq!(planned_path("ij->ji", &[&[2, 3]], EinsumPathStrategy::Explicit(Vec::new())), vec![]);
        let shapes: &[&[usize]] = &[&[2, 3], &[3, 4]];
        assert_eq!(planned_path("ij,jk", shapes, EinsumPathStrategy::Explicit(vec![(1, 0)])), vec![(1, 0)]);
        let types = shapes.iter().map(|shape| ArrayType::new_static(DataType::F32, shape.to_vec())).collect::<Vec<_>>();
        assert_eq!(
            "ij,jk"
                .parse::<EinsumSpecification>()
                .unwrap()
                .plan(&types, &EinsumOptions::new().with_path(EinsumPathStrategy::Explicit(Vec::new())))
                .map(|_| ()),
            Err(ProgramError::Type(TypeError::invalid(
                "`einsum` path must have 1 pairwise contractions for 2 inputs but has 0",
            ))),
        );
    }

    #[test]
    fn test_einsum_path_strategy_matches_opt_einsum() {
        // These paths were recorded from `opt_einsum.contract_path` 3.4.0, which `jax.numpy.einsum` calls, for a
        // seeded random corpus of subscripts and shapes. They cover every search that the automatic strategy selects
        // (i.e., three to eighteen inputs), implicit outputs, repeated labels, and widely different extents, which
        // exercise the pruning of the branching searches.
        let auto: &[(&str, &[&[usize]], &[(usize, usize)])] = &[
            ("c,bac,b->", &[&[3], &[5, 3, 3], &[5]], &[(1, 2), (0, 1)]),
            ("igdf,dih,id->", &[&[9, 9, 9, 2], &[9, 9, 9], &[9, 9]], &[(0, 2), (0, 1)]),
            ("fg,hf,ceab", &[&[8, 8], &[8, 8], &[8, 8, 8, 8]], &[(0, 1), (0, 1)]),
            ("bd,bee,cdea->dbe", &[&[11, 11], &[11, 3, 3], &[5, 11, 3, 5]], &[(0, 1), (0, 1)]),
            ("f,bdcf,ef,debc->cde", &[&[5], &[2, 4, 5, 5], &[3, 5], &[4, 3, 2, 5]], &[(0, 2), (0, 2), (0, 1)]),
            ("fcg,b,chda,d->gc", &[&[64, 64, 2], &[2], &[64, 64, 2, 9], &[2]], &[(1, 3), (1, 2), (0, 1)]),
            ("f,a,cad,baf", &[&[8], &[8], &[8, 8, 8], &[8, 8, 8]], &[(0, 1), (1, 2), (0, 1)]),
            ("cabd,acd,e,bea->aec", &[&[3, 11, 5, 300], &[11, 3, 300], &[2], &[5, 2, 11]], &[(0, 1), (1, 2), (0, 1)]),
            ("ca,cde,b,c,dcb->de", &[&[5, 4], &[5, 4, 3], &[5], &[5], &[4, 5, 5]], &[(0, 3), (1, 2), (1, 2), (0, 1)]),
            ("df,abec,df,e,e->bc", &[&[9, 64], &[2, 9, 2, 2], &[9, 64], &[2], &[2]], &[(0, 2), (1, 2), (0, 2), (0, 1)]),
            (
                "acb,a,bac,bca,ac",
                &[&[8, 8, 8], &[8], &[8, 8, 8], &[8, 8, 8], &[8, 8]],
                &[(0, 2), (1, 3), (0, 1), (0, 1)],
            ),
            ("cb,b,bc,aac,aa->bac", &[&[3, 2], &[2], &[2, 3], &[3, 3, 3], &[3, 3]], &[(0, 2), (1, 2), (0, 1), (0, 1)]),
            (
                "gcae,ca,adgf,d,cegb,becg->f",
                &[&[4, 5, 3, 5], &[5, 3], &[3, 4, 4, 4], &[4], &[5, 5, 4, 4], &[4, 5, 5, 4]],
                &[(4, 5), (0, 4), (0, 3), (0, 2), (0, 1)],
            ),
            (
                "gfh,dage,bc,gbae,a,fca->",
                &[&[2, 9, 2], &[64, 2, 2, 9], &[64, 64], &[2, 64, 2, 9], &[2], &[9, 64, 2]],
                &[(1, 4), (2, 4), (1, 3), (1, 2), (0, 1)],
            ),
            (
                "gf,egih,cadb,d,haf,cbgf",
                &[&[8, 8], &[8, 8, 8, 8], &[8, 8, 8, 8], &[8], &[8, 8, 8], &[8, 8, 8, 8]],
                &[(2, 3), (0, 3), (2, 3), (1, 2), (0, 1)],
            ),
            (
                "bcca,c,abc,aab,cc,bac->cb",
                &[&[5, 300, 300, 7], &[300], &[7, 5, 300], &[7, 7, 5], &[300, 300], &[5, 7, 300]],
                &[(0, 2), (3, 4), (0, 2), (1, 2), (0, 1)],
            ),
            (
                "ca,b,ac,c,c,a,cba->ba",
                &[&[4, 2], &[5], &[2, 4], &[4], &[4], &[2], &[4, 5, 2]],
                &[(3, 4), (0, 2), (0, 2), (1, 2), (0, 2), (0, 1)],
            ),
            (
                "adbc,acd,bca,c,bdca,cb,dacb->dcb",
                &[&[9, 64, 2, 9], &[9, 9, 64], &[2, 9, 9], &[9], &[2, 64, 9, 9], &[9, 2], &[64, 9, 9, 2]],
                &[(0, 4), (4, 5), (0, 4), (0, 2), (0, 2), (0, 1)],
            ),
            (
                "dac,bcde,ecda,ed,a,cbed,daeb",
                &[&[8, 8, 8], &[8, 8, 8, 8], &[8, 8, 8, 8], &[8, 8], &[8], &[8, 8, 8, 8], &[8, 8, 8, 8]],
                &[(0, 4), (0, 3), (0, 3), (0, 2), (0, 2), (0, 1)],
            ),
            (
                "adbe,ecd,aa,edc,db,dca,eab->b",
                &[&[2, 300, 300, 5], &[5, 7, 300], &[2, 2], &[5, 300, 7], &[300, 300], &[300, 7, 2], &[5, 2, 300]],
                &[(1, 3), (3, 5), (0, 4), (2, 3), (0, 2), (0, 1)],
            ),
            (
                "ecda,e,edc,c,ch,g,hgf,ea->",
                &[&[5, 5, 3, 4], &[5], &[5, 3, 5], &[5], &[5, 5], &[5], &[5, 5, 4], &[5, 4]],
                &[(0, 7), (1, 6), (3, 4), (0, 3), (1, 2), (0, 1), (0, 1)],
            ),
            (
                "cab,bca,c,b,acb,ab,bc,b->bca",
                &[&[64, 2, 9], &[9, 64, 2], &[64], &[9], &[2, 64, 9], &[2, 9], &[9, 64], &[9]],
                &[(0, 1), (2, 6), (0, 3), (3, 4), (0, 2), (0, 2), (0, 1)],
            ),
            (
                "hd,c,id,igf,fhb,bc,afi,e",
                &[&[8, 8], &[8], &[8, 8], &[8, 8, 8], &[8, 8, 8], &[8, 8], &[8, 8, 8], &[8]],
                &[(1, 5), (3, 6), (0, 1), (3, 4), (0, 3), (0, 2), (0, 1)],
            ),
            (
                "bbca,acb,bb,aab,bacc,ba,ca,cc->c",
                &[&[7, 7, 3, 3], &[3, 3, 7], &[7, 7], &[3, 3, 7], &[7, 3, 3, 3], &[7, 3], &[3, 3], &[3, 3]],
                &[(0, 1), (2, 6), (1, 2), (3, 4), (0, 3), (0, 1), (0, 1)],
            ),
            (
                "cdab,ac,bdca,ca,dcba,a,abdc,bad,b->d",
                &[&[4, 5, 3, 4], &[3, 4], &[4, 5, 4, 3], &[4, 3], &[5, 4, 4, 3], &[3], &[3, 4, 5, 4], &[4, 3, 5], &[4]],
                &[(0, 2), (2, 4), (5, 6), (3, 5), (0, 1), (2, 3), (1, 2), (0, 1)],
            ),
            (
                "h,ghe,c,dbif,gicd,ice,dgh,c,bch->e",
                &[&[2], &[2, 2, 9], &[64], &[9, 2, 9, 9], &[2, 9, 64, 9], &[9, 64, 9], &[9, 2, 2], &[64], &[2, 64, 2]],
                &[(2, 7), (6, 7), (0, 1), (0, 3), (0, 4), (0, 3), (0, 2), (0, 1)],
            ),
            (
                "gdf,eabg,c,d,gc,dcb,e,db,abfc",
                &[&[8, 8, 8], &[8, 8, 8, 8], &[8], &[8], &[8, 8], &[8, 8, 8], &[8], &[8, 8], &[8, 8, 8, 8]],
                &[(2, 4), (2, 5), (1, 3), (1, 4), (0, 2), (0, 1), (1, 2), (0, 1)],
            ),
            (
                "bb,abdc,abadc,cdba,addc,ba,abdd,cc,dc->",
                &[
                    &[3, 3],
                    &[5, 3, 3, 300],
                    &[5, 3, 5, 3, 300],
                    &[300, 3, 3, 5],
                    &[5, 3, 3, 300],
                    &[3, 5],
                    &[5, 3, 3, 3],
                    &[300, 300],
                    &[3, 300],
                ],
                &[(5, 6), (0, 7), (0, 1), (0, 5), (0, 4), (1, 3), (0, 2), (0, 1)],
            ),
            (
                "abd,adbc,ea,de,fdbe,c,e,afdc,ed,c->fd",
                &[&[2, 3, 4], &[2, 4, 3, 3], &[3, 2], &[4, 3], &[5, 4, 3, 3], &[3], &[3], &[2, 5, 4, 3], &[3, 4], &[3]],
                &[(0, 1), (1, 6), (1, 7), (1, 4), (0, 1), (2, 4), (1, 3), (0, 2), (0, 1)],
            ),
            (
                "b,bdac,bc,ab,cd,d,db,cd,bac,bacd->a",
                &[&[2], &[2, 9, 2, 2], &[2, 2], &[2, 2], &[2, 9], &[9], &[9, 2], &[2, 9], &[2, 2, 2], &[2, 2, 2, 9]],
                &[(2, 8), (3, 6), (1, 5), (3, 6), (4, 5), (2, 4), (2, 3), (1, 2), (0, 1)],
            ),
            (
                "bca,a,bca,acb,acb,abc,ca,cab,bc,c",
                &[&[8, 8, 8], &[8], &[8, 8, 8], &[8, 8, 8], &[8, 8, 8], &[8, 8, 8], &[8, 8], &[8, 8, 8], &[8, 8], &[8]],
                &[(0, 2), (1, 2), (1, 3), (4, 5), (4, 5), (1, 4), (1, 3), (0, 2), (0, 1)],
            ),
            (
                "dd,c,bbde,ebda,cda,edc,ad,caeb,aadb,cee->e",
                &[
                    &[7, 7],
                    &[11],
                    &[11, 11, 7, 3],
                    &[3, 11, 7, 11],
                    &[11, 7, 11],
                    &[3, 7, 11],
                    &[11, 7],
                    &[11, 11, 3, 11],
                    &[11, 11, 7, 11],
                    &[11, 3, 3],
                ],
                &[(1, 9), (4, 8), (3, 4), (2, 4), (1, 5), (0, 2), (1, 3), (0, 2), (0, 1)],
            ),
            (
                "bca,bc,cb,ac,c,b,b,ac,c,bac,ac->cb",
                &[&[3, 3, 2], &[3, 3], &[3, 3], &[2, 3], &[3], &[3], &[3], &[2, 3], &[3], &[3, 2, 3], &[2, 3]],
                &[(0, 9), (0, 1), (7, 8), (0, 4), (4, 6), (4, 5), (0, 3), (0, 1), (0, 1), (0, 1)],
            ),
            (
                "g,cbge,bd,f,d,agd,gbe,f,agd,daec,bag->gc",
                &[
                    &[64],
                    &[64, 9, 64, 9],
                    &[9, 9],
                    &[9],
                    &[9],
                    &[64, 64, 9],
                    &[64, 9, 9],
                    &[9],
                    &[64, 64, 9],
                    &[9, 64, 9, 64],
                    &[9, 64, 64],
                ],
                &[(0, 6), (0, 9), (3, 5), (1, 3), (0, 1), (1, 3), (3, 4), (0, 3), (0, 2), (0, 1)],
            ),
            (
                "cd,c,ad,cbha,cha,gea,fec,hce,abfd,gdbe,dfgb",
                &[
                    &[8, 8],
                    &[8],
                    &[8, 8],
                    &[8, 8, 8, 8],
                    &[8, 8, 8],
                    &[8, 8, 8],
                    &[8, 8, 8],
                    &[8, 8, 8],
                    &[8, 8, 8, 8],
                    &[8, 8, 8, 8],
                    &[8, 8, 8, 8],
                ],
                &[(0, 1), (0, 6), (0, 1), (1, 2), (1, 2), (0, 5), (1, 4), (1, 2), (1, 2), (0, 1)],
            ),
            (
                "gaac,a,edg,ahbf,ddea,hcg,hcef,ff,h,fh,gc->bc",
                &[
                    &[3, 11, 11, 2],
                    &[11],
                    &[5, 11, 3],
                    &[11, 7, 5, 2],
                    &[11, 11, 5, 11],
                    &[7, 2, 3],
                    &[7, 2, 5, 2],
                    &[2, 2],
                    &[7],
                    &[2, 7],
                    &[3, 2],
                ],
                &[(6, 9), (0, 1), (0, 2), (1, 3), (2, 6), (1, 2), (1, 2), (1, 2), (1, 2), (0, 1)],
            ),
            (
                "d,cdbe,d,baec,ab,e,aeb,acd,ac,e,a,cea->adc",
                &[
                    &[3],
                    &[3, 3, 4, 2],
                    &[3],
                    &[4, 5, 2, 3],
                    &[5, 4],
                    &[2],
                    &[5, 2, 4],
                    &[5, 3, 3],
                    &[5, 3],
                    &[2],
                    &[5],
                    &[3, 2, 5],
                ],
                &[(0, 2), (1, 4), (7, 9), (1, 8), (2, 3), (3, 6), (3, 5), (1, 2), (0, 3), (0, 2), (0, 1)],
            ),
            (
                "fecb,fe,ebf,d,cd,ab,f,f,bdf,faed,cb,dbf->fab",
                &[
                    &[9, 64, 2, 9],
                    &[9, 64],
                    &[64, 9, 9],
                    &[64],
                    &[2, 64],
                    &[9, 9],
                    &[9],
                    &[9],
                    &[9, 64, 9],
                    &[9, 9, 64, 64],
                    &[2, 9],
                    &[64, 9, 9],
                ],
                &[(0, 2), (6, 9), (0, 8), (0, 1), (4, 6), (1, 2), (2, 5), (2, 4), (0, 2), (0, 2), (0, 1)],
            ),
            (
                "ade,abd,f,b,bdg,bcfg,degb,fcbe,cef,afg,bad,da",
                &[
                    &[8, 8, 8],
                    &[8, 8, 8],
                    &[8],
                    &[8],
                    &[8, 8, 8],
                    &[8, 8, 8, 8],
                    &[8, 8, 8, 8],
                    &[8, 8, 8, 8],
                    &[8, 8, 8],
                    &[8, 8, 8],
                    &[8, 8, 8],
                    &[8, 8],
                ],
                &[(1, 10), (0, 9), (2, 4), (3, 4), (0, 3), (0, 2), (1, 5), (1, 4), (2, 3), (0, 2), (0, 1)],
            ),
            (
                "c,dabc,daac,c,ac,dab,dac,dd,cad,bc,cbdd,adac->c",
                &[
                    &[7],
                    &[300, 2, 11, 7],
                    &[300, 2, 2, 7],
                    &[7],
                    &[2, 7],
                    &[300, 2, 11],
                    &[300, 2, 7],
                    &[300, 300],
                    &[7, 2, 300],
                    &[11, 7],
                    &[7, 11, 300, 300],
                    &[2, 300, 2, 7],
                ],
                &[(0, 3), (0, 8), (2, 9), (0, 2), (2, 4), (5, 6), (4, 5), (1, 4), (1, 3), (0, 2), (0, 1)],
            ),
            (
                "acdb,fb,cde,e,fced,eda,eaf,c,fc,dafc,aecf,af,fc->bda",
                &[
                    &[3, 5, 5, 3],
                    &[4, 3],
                    &[5, 5, 5],
                    &[5],
                    &[4, 5, 5, 5],
                    &[5, 5, 3],
                    &[5, 3, 4],
                    &[5],
                    &[4, 5],
                    &[5, 3, 4, 5],
                    &[3, 5, 5, 4],
                    &[3, 4],
                    &[4, 5],
                ],
                &[(8, 12), (6, 10), (8, 10), (2, 4), (5, 6), (2, 3), (0, 2), (3, 5), (0, 4), (0, 1), (1, 2), (0, 1)],
            ),
            (
                "d,bdg,ca,afg,c,gef,c,fa,gea,gad,gdce,ab,c->cf",
                &[
                    &[2],
                    &[64, 2, 64],
                    &[64, 64],
                    &[64, 2, 64],
                    &[64],
                    &[64, 64, 2],
                    &[64],
                    &[2, 64],
                    &[64, 64, 64],
                    &[64, 64, 2],
                    &[64, 2, 64, 64],
                    &[64, 64],
                    &[64],
                ],
                &[(3, 7), (3, 5), (8, 10), (2, 9), (0, 1), (4, 7), (2, 6), (3, 5), (0, 1), (2, 3), (0, 2), (0, 1)],
            ),
            (
                "dcai,cb,bgc,hd,h,h,ch,degc,a,ci,ai,hai,hc",
                &[
                    &[8, 8, 8, 8],
                    &[8, 8],
                    &[8, 8, 8],
                    &[8, 8],
                    &[8],
                    &[8],
                    &[8, 8],
                    &[8, 8, 8, 8],
                    &[8],
                    &[8, 8],
                    &[8, 8],
                    &[8, 8, 8],
                    &[8, 8],
                ],
                &[(1, 2), (4, 10), (7, 8), (0, 6), (3, 5), (1, 2), (0, 6), (0, 2), (0, 3), (1, 3), (0, 1), (0, 1)],
            ),
            (
                "ceb,eba,bb,abeed,dabe,aeb,bcd,beac,ebb,eda,acb,abcda,bbde->",
                &[
                    &[300, 11, 300],
                    &[11, 300, 5],
                    &[300, 300],
                    &[5, 300, 11, 11, 300],
                    &[300, 5, 300, 11],
                    &[5, 11, 300],
                    &[300, 300, 300],
                    &[300, 11, 5, 300],
                    &[11, 300, 300],
                    &[11, 300, 5],
                    &[5, 300, 300],
                    &[5, 300, 300, 300, 5],
                    &[300, 300, 300, 11],
                ],
                &[(1, 5), (6, 11), (1, 10), (1, 2), (0, 2), (4, 6), (2, 5), (1, 4), (2, 3), (0, 1), (0, 2), (0, 1)],
            ),
            (
                "ba,cb,a,ac,cab,cab,c,cba,bca,a,a,a,cba,c->",
                &[
                    &[3, 2],
                    &[3, 3],
                    &[2],
                    &[2, 3],
                    &[3, 2, 3],
                    &[3, 2, 3],
                    &[3],
                    &[3, 3, 2],
                    &[3, 3, 2],
                    &[2],
                    &[2],
                    &[2],
                    &[3, 3, 2],
                    &[3],
                ],
                &[
                    (1, 4),
                    (3, 5),
                    (4, 8),
                    (8, 9),
                    (8, 9),
                    (0, 8),
                    (1, 7),
                    (1, 5),
                    (4, 5),
                    (0, 1),
                    (0, 1),
                    (1, 2),
                    (0, 1),
                ],
            ),
            (
                "edcb,bdce,b,bad,ce,d,dabe,ceb,dbea,c,eba,ce,ad,edb->",
                &[
                    &[9, 64, 9, 9],
                    &[9, 64, 9, 9],
                    &[9],
                    &[9, 2, 64],
                    &[9, 9],
                    &[64],
                    &[64, 2, 9, 9],
                    &[9, 9, 9],
                    &[64, 9, 9, 2],
                    &[9],
                    &[9, 9, 2],
                    &[9, 9],
                    &[2, 64],
                    &[9, 64, 9],
                ],
                &[
                    (0, 1),
                    (4, 6),
                    (9, 11),
                    (1, 10),
                    (3, 8),
                    (4, 7),
                    (5, 7),
                    (1, 4),
                    (3, 5),
                    (1, 3),
                    (0, 3),
                    (0, 1),
                    (0, 1),
                ],
            ),
            (
                "abec,bac,dc,be,abce,bedc,cbed,debc,ad,acde,dba,ecd,ed,cab",
                &[
                    &[8, 8, 8, 8],
                    &[8, 8, 8],
                    &[8, 8],
                    &[8, 8],
                    &[8, 8, 8, 8],
                    &[8, 8, 8, 8],
                    &[8, 8, 8, 8],
                    &[8, 8, 8, 8],
                    &[8, 8],
                    &[8, 8, 8, 8],
                    &[8, 8, 8],
                    &[8, 8, 8],
                    &[8, 8],
                    &[8, 8, 8],
                ],
                &[
                    (0, 4),
                    (3, 4),
                    (3, 11),
                    (0, 8),
                    (3, 5),
                    (5, 7),
                    (2, 3),
                    (0, 3),
                    (0, 3),
                    (0, 1),
                    (1, 2),
                    (1, 2),
                    (0, 1),
                ],
            ),
            (
                "ac,acb,aac,a,bbca,ab,c,b,b,b,abc,cab,aca,ba->ba",
                &[
                    &[7, 300],
                    &[7, 300, 11],
                    &[7, 7, 300],
                    &[7],
                    &[11, 11, 300, 7],
                    &[7, 11],
                    &[300],
                    &[11],
                    &[11],
                    &[11],
                    &[7, 11, 300],
                    &[300, 7, 11],
                    &[7, 300, 7],
                    &[11, 7],
                ],
                &[
                    (0, 2),
                    (0, 2),
                    (6, 7),
                    (9, 10),
                    (6, 8),
                    (7, 8),
                    (2, 7),
                    (1, 5),
                    (4, 5),
                    (1, 2),
                    (1, 3),
                    (1, 2),
                    (0, 1),
                ],
            ),
            (
                "bag,gd,gcb,fbec,adg,a,edca,e,bcdg,cedg,bde,db,f,dbe,ea->gef",
                &[
                    &[2, 5, 5],
                    &[5, 5],
                    &[5, 4, 2],
                    &[4, 2, 5, 4],
                    &[5, 5, 5],
                    &[5],
                    &[5, 5, 4, 5],
                    &[5],
                    &[2, 4, 5, 5],
                    &[4, 5, 5, 5],
                    &[2, 5, 5],
                    &[5, 2],
                    &[4],
                    &[5, 2, 5],
                    &[5, 5],
                ],
                &[
                    (10, 13),
                    (2, 8),
                    (1, 3),
                    (3, 8),
                    (5, 8),
                    (0, 2),
                    (5, 8),
                    (5, 7),
                    (2, 6),
                    (4, 5),
                    (3, 4),
                    (0, 3),
                    (0, 1),
                    (0, 1),
                ],
            ),
            (
                "ibe,hbeg,hfgb,gi,ahg,g,dgb,gdi,f,fgd,dhaf,dh,hfd,ghcd,g->",
                &[
                    &[2, 9, 64],
                    &[2, 9, 64, 64],
                    &[2, 64, 64, 9],
                    &[64, 2],
                    &[2, 2, 64],
                    &[64],
                    &[9, 64, 9],
                    &[64, 9, 2],
                    &[64],
                    &[64, 64, 9],
                    &[9, 2, 2, 64],
                    &[9, 2],
                    &[2, 64, 9],
                    &[64, 2, 2, 9],
                    &[64],
                ],
                &[
                    (5, 14),
                    (0, 1),
                    (10, 11),
                    (7, 9),
                    (1, 4),
                    (0, 3),
                    (3, 5),
                    (5, 7),
                    (0, 6),
                    (2, 5),
                    (0, 4),
                    (2, 3),
                    (0, 2),
                    (0, 1),
                ],
            ),
            (
                "gea,d,a,c,dea,f,dcga,ge,fcea,fged,c,cef,geab,aedg,cb",
                &[
                    &[8, 8, 8],
                    &[8],
                    &[8],
                    &[8],
                    &[8, 8, 8],
                    &[8],
                    &[8, 8, 8, 8],
                    &[8, 8],
                    &[8, 8, 8, 8],
                    &[8, 8, 8, 8],
                    &[8],
                    &[8, 8, 8],
                    &[8, 8, 8, 8],
                    &[8, 8, 8, 8],
                    &[8, 8],
                ],
                &[
                    (3, 10),
                    (7, 9),
                    (0, 8),
                    (2, 7),
                    (4, 5),
                    (4, 7),
                    (2, 5),
                    (2, 3),
                    (1, 5),
                    (0, 2),
                    (0, 2),
                    (0, 3),
                    (0, 2),
                    (0, 1),
                ],
            ),
            (
                "cbd,bacae,ce,dde,c,cde,a,dba,dbec,aed,eebc,c,d,bde,eb->",
                &[
                    &[3, 7, 5],
                    &[7, 11, 3, 11, 3],
                    &[3, 3],
                    &[5, 5, 3],
                    &[3],
                    &[3, 5, 3],
                    &[11],
                    &[5, 7, 11],
                    &[5, 7, 3, 3],
                    &[11, 3, 5],
                    &[3, 3, 7, 3],
                    &[3],
                    &[5],
                    &[7, 5, 3],
                    &[3, 7],
                ],
                &[
                    (4, 11),
                    (0, 7),
                    (9, 12),
                    (0, 7),
                    (2, 9),
                    (6, 8),
                    (2, 8),
                    (0, 7),
                    (1, 3),
                    (2, 4),
                    (2, 4),
                    (2, 3),
                    (1, 2),
                    (0, 1),
                ],
            ),
            (
                "b,edg,a,f,e,ba,bcea,abfe,cgb,a,d,fed,ea,bcde,c,cg->",
                &[
                    &[4],
                    &[5, 2, 4],
                    &[4],
                    &[4],
                    &[5],
                    &[4, 4],
                    &[4, 3, 5, 4],
                    &[4, 4, 4, 5],
                    &[3, 4, 4],
                    &[4],
                    &[2],
                    &[4, 5, 2],
                    &[5, 4],
                    &[4, 3, 2, 5],
                    &[3],
                    &[3, 4],
                ],
                &[
                    (2, 9),
                    (5, 10),
                    (4, 13),
                    (5, 10),
                    (1, 3),
                    (1, 2),
                    (2, 9),
                    (0, 6),
                    (2, 4),
                    (2, 6),
                    (3, 5),
                    (1, 4),
                    (2, 3),
                    (1, 2),
                    (0, 1),
                ],
            ),
            (
                "gbf,geb,h,d,ceia,fib,ehfg,ec,bedi,hi,a,dhgb,aife,d,i,hif->hd",
                &[
                    &[2, 9, 9],
                    &[2, 9, 9],
                    &[9],
                    &[64],
                    &[2, 9, 9, 64],
                    &[9, 9, 9],
                    &[9, 9, 9, 2],
                    &[9, 2],
                    &[9, 9, 64, 9],
                    &[9, 9],
                    &[64],
                    &[64, 9, 2, 9],
                    &[64, 9, 9, 9],
                    &[64],
                    &[9],
                    &[9, 9, 9],
                ],
                &[
                    (3, 13),
                    (3, 6),
                    (9, 13),
                    (7, 12),
                    (6, 9),
                    (3, 7),
                    (0, 9),
                    (2, 7),
                    (5, 7),
                    (5, 6),
                    (0, 5),
                    (1, 4),
                    (1, 3),
                    (0, 1),
                    (0, 1),
                ],
            ),
            (
                "ebca,dacb,acd,ec,cdeb,d,e,be,eacb,cda,ebdc,ce,bead,eadc,eb,dcae",
                &[
                    &[8, 8, 8, 8],
                    &[8, 8, 8, 8],
                    &[8, 8, 8],
                    &[8, 8],
                    &[8, 8, 8, 8],
                    &[8],
                    &[8],
                    &[8, 8],
                    &[8, 8, 8, 8],
                    &[8, 8, 8],
                    &[8, 8, 8, 8],
                    &[8, 8],
                    &[8, 8, 8, 8],
                    &[8, 8, 8, 8],
                    &[8, 8],
                    &[8, 8, 8, 8],
                ],
                &[
                    (0, 8),
                    (1, 7),
                    (2, 6),
                    (1, 5),
                    (3, 6),
                    (4, 5),
                    (0, 5),
                    (3, 5),
                    (2, 4),
                    (0, 6),
                    (0, 5),
                    (2, 4),
                    (0, 3),
                    (0, 2),
                    (0, 1),
                ],
            ),
            (
                "hbd,h,e,bc,febc,dbc,ff,dd,ec,f,bga,bec,ff,aa,fbfd,dbg->bg",
                &[
                    &[7, 300, 11],
                    &[7],
                    &[2],
                    &[300, 3],
                    &[300, 2, 300, 3],
                    &[11, 300, 3],
                    &[300, 300],
                    &[11, 11],
                    &[2, 3],
                    &[300],
                    &[300, 300, 11],
                    &[300, 2, 3],
                    &[300, 300],
                    &[11, 11],
                    &[300, 300, 300, 11],
                    &[11, 300, 300],
                ],
                &[
                    (6, 9),
                    (10, 14),
                    (8, 10),
                    (0, 1),
                    (3, 11),
                    (2, 5),
                    (1, 9),
                    (5, 8),
                    (3, 7),
                    (5, 6),
                    (2, 5),
                    (0, 4),
                    (1, 3),
                    (0, 2),
                    (0, 1),
                ],
            ),
            (
                "bg,b,e,dfg,gecf,ag,acg,gh,gdef,gf,gafc,hed,bcg,fdg,fh,h,bhgc->",
                &[
                    &[3, 3],
                    &[3],
                    &[2],
                    &[4, 3, 3],
                    &[3, 2, 3, 3],
                    &[2, 3],
                    &[2, 3, 3],
                    &[3, 4],
                    &[3, 4, 2, 3],
                    &[3, 3],
                    &[3, 2, 3, 3],
                    &[4, 2, 4],
                    &[3, 3, 3],
                    &[3, 4, 3],
                    &[3, 4],
                    &[4],
                    &[3, 4, 3, 3],
                ],
                &[
                    (3, 13),
                    (7, 15),
                    (9, 14),
                    (9, 12),
                    (5, 8),
                    (4, 11),
                    (3, 10),
                    (3, 8),
                    (3, 7),
                    (0, 6),
                    (0, 6),
                    (4, 5),
                    (3, 4),
                    (1, 3),
                    (1, 2),
                    (0, 1),
                ],
            ),
            (
                "abcf,dba,f,acfd,acb,dfc,df,ecda,caf,cbd,dcaf,ed,fdae,edfb,b,ca,dc->ead",
                &[
                    &[64, 64, 64, 9],
                    &[9, 64, 64],
                    &[9],
                    &[64, 64, 9, 9],
                    &[64, 64, 64],
                    &[9, 9, 64],
                    &[9, 9],
                    &[9, 64, 9, 64],
                    &[64, 64, 9],
                    &[64, 64, 9],
                    &[9, 64, 64, 9],
                    &[9, 9],
                    &[9, 9, 64, 9],
                    &[9, 9, 9, 64],
                    &[64],
                    &[64, 64],
                    &[9, 64],
                ],
                &[
                    (3, 10),
                    (0, 3),
                    (5, 14),
                    (2, 12),
                    (3, 9),
                    (3, 8),
                    (2, 4),
                    (0, 4),
                    (0, 7),
                    (5, 6),
                    (3, 5),
                    (3, 5),
                    (2, 3),
                    (2, 3),
                    (1, 2),
                    (0, 1),
                ],
            ),
            (
                "cb,cba,bca,b,c,c,bac,cab,abc,cab,ab,abc,cab,cab,ca,bc,ac",
                &[
                    &[8, 8],
                    &[8, 8, 8],
                    &[8, 8, 8],
                    &[8],
                    &[8],
                    &[8],
                    &[8, 8, 8],
                    &[8, 8, 8],
                    &[8, 8, 8],
                    &[8, 8, 8],
                    &[8, 8],
                    &[8, 8, 8],
                    &[8, 8, 8],
                    &[8, 8, 8],
                    &[8, 8],
                    &[8, 8],
                    &[8, 8],
                ],
                &[
                    (1, 2),
                    (2, 3),
                    (2, 13),
                    (2, 13),
                    (2, 12),
                    (2, 11),
                    (3, 10),
                    (3, 9),
                    (3, 8),
                    (0, 4),
                    (2, 3),
                    (1, 3),
                    (3, 4),
                    (2, 3),
                    (0, 2),
                    (0, 1),
                ],
            ),
            (
                "a,h,aih,h,chbf,bdg,gfigb,if,gei,g,igbc,c,g,fbc,aa,g,cegfe->fe",
                &[
                    &[2],
                    &[5],
                    &[2, 11, 5],
                    &[5],
                    &[7, 5, 2, 300],
                    &[2, 2, 2],
                    &[2, 300, 11, 2, 2],
                    &[11, 300],
                    &[2, 300, 11],
                    &[2],
                    &[11, 2, 2, 7],
                    &[7],
                    &[2],
                    &[300, 2, 7],
                    &[2, 2],
                    &[2],
                    &[7, 300, 2, 300, 300],
                ],
                &[
                    (1, 3),
                    (7, 10),
                    (0, 10),
                    (9, 12),
                    (1, 8),
                    (2, 3),
                    (0, 7),
                    (0, 8),
                    (1, 2),
                    (2, 4),
                    (2, 4),
                    (2, 4),
                    (3, 4),
                    (2, 3),
                    (1, 2),
                    (0, 1),
                ],
            ),
            (
                "gc,ebaf,ef,bcdf,dh,ceba,c,eafd,egf,eghf,e,aehb,bhe,bfhe,fbd,c,gb,fe->",
                &[
                    &[2, 4],
                    &[2, 5, 5, 4],
                    &[2, 4],
                    &[5, 4, 2, 4],
                    &[2, 3],
                    &[4, 2, 5, 5],
                    &[4],
                    &[2, 5, 4, 2],
                    &[2, 2, 4],
                    &[2, 2, 3, 4],
                    &[2],
                    &[5, 2, 3, 5],
                    &[5, 3, 2],
                    &[5, 4, 3, 2],
                    &[4, 5, 2],
                    &[4],
                    &[2, 5],
                    &[4, 2],
                ],
                &[
                    (6, 15),
                    (2, 15),
                    (2, 12),
                    (8, 9),
                    (5, 6),
                    (1, 9),
                    (0, 7),
                    (1, 3),
                    (3, 8),
                    (2, 5),
                    (1, 4),
                    (1, 4),
                    (1, 2),
                    (2, 4),
                    (2, 3),
                    (1, 2),
                    (0, 1),
                ],
            ),
            (
                "ed,bdec,cad,bade,ae,cad,bc,dca,cbde,dbea,eacd,ab,d,c,ebc,bdac,bcea,c->",
                &[
                    &[2, 9],
                    &[2, 9, 2, 64],
                    &[64, 9, 9],
                    &[2, 9, 9, 2],
                    &[9, 2],
                    &[64, 9, 9],
                    &[2, 64],
                    &[9, 64, 9],
                    &[64, 2, 9, 2],
                    &[9, 2, 2, 9],
                    &[2, 9, 64, 9],
                    &[9, 2],
                    &[9],
                    &[64],
                    &[2, 2, 64],
                    &[2, 9, 9, 64],
                    &[2, 64, 2, 9],
                    &[64],
                ],
                &[
                    (2, 5),
                    (5, 16),
                    (1, 5),
                    (1, 4),
                    (6, 10),
                    (3, 9),
                    (5, 7),
                    (2, 10),
                    (7, 9),
                    (0, 7),
                    (0, 7),
                    (0, 2),
                    (0, 5),
                    (3, 4),
                    (0, 3),
                    (1, 2),
                    (0, 1),
                ],
            ),
            (
                "edfa,ac,cfba,badf,ebaf,abef,afbe,abdf,af,fadb,afbe,aecb,a,d,fdcb,bfac,de,adcf",
                &[
                    &[8, 8, 8, 8],
                    &[8, 8],
                    &[8, 8, 8, 8],
                    &[8, 8, 8, 8],
                    &[8, 8, 8, 8],
                    &[8, 8, 8, 8],
                    &[8, 8, 8, 8],
                    &[8, 8, 8, 8],
                    &[8, 8],
                    &[8, 8, 8, 8],
                    &[8, 8, 8, 8],
                    &[8, 8, 8, 8],
                    &[8],
                    &[8],
                    &[8, 8, 8, 8],
                    &[8, 8, 8, 8],
                    &[8, 8],
                    &[8, 8, 8, 8],
                ],
                &[
                    (4, 5),
                    (4, 16),
                    (3, 4),
                    (4, 14),
                    (4, 12),
                    (2, 8),
                    (0, 2),
                    (0, 1),
                    (3, 8),
                    (0, 7),
                    (0, 1),
                    (0, 6),
                    (0, 5),
                    (1, 4),
                    (1, 3),
                    (0, 2),
                    (0, 1),
                ],
            ),
            (
                "dcb,bdcaa,bb,bdac,cdd,ba,d,bdac,bcdab,cd,d,a,aadcb,dbc,a,c,dcab,b->adb",
                &[
                    &[300, 300, 5],
                    &[5, 300, 300, 5, 5],
                    &[5, 5],
                    &[5, 300, 5, 300],
                    &[300, 300, 300],
                    &[5, 5],
                    &[300],
                    &[5, 300, 5, 300],
                    &[5, 300, 300, 5, 5],
                    &[300, 300],
                    &[300],
                    &[5],
                    &[5, 5, 300, 300, 5],
                    &[300, 5, 300],
                    &[5],
                    &[300],
                    &[300, 300, 5, 5],
                    &[5],
                ],
                &[
                    (1, 3),
                    (5, 16),
                    (5, 15),
                    (2, 5),
                    (3, 4),
                    (4, 10),
                    (0, 4),
                    (2, 3),
                    (3, 7),
                    (0, 3),
                    (4, 6),
                    (2, 6),
                    (1, 5),
                    (2, 3),
                    (0, 3),
                    (0, 2),
                    (0, 1),
                ],
            ),
        ];
        let optimal: &[(&str, &[&[usize]], &[(usize, usize)])] = &[
            ("bda,cadb,dbc->adb", &[&[4, 10, 10], &[3, 10, 10, 4], &[10, 4, 3]], &[(1, 2), (0, 1)]),
            ("bg,fgb,ecbg->gcf", &[&[2, 3], &[2, 3, 2], &[10, 2, 2, 3]], &[(0, 2), (0, 1)]),
            ("bf,dgcb,gbaf->bf", &[&[10, 2], &[2, 3, 10, 10], &[3, 10, 10, 2]], &[(0, 1), (0, 1)]),
            ("a,a,dbac,dac->", &[&[2], &[2], &[10, 10, 2, 2], &[10, 2, 2]], &[(0, 1), (0, 2), (0, 1)]),
            ("b,c,c,bc->b", &[&[2], &[2], &[2], &[2, 2]], &[(1, 2), (0, 1), (0, 1)]),
            ("be,bdc,de,dc->", &[&[4, 2], &[4, 10, 10], &[10, 2], &[10, 10]], &[(1, 3), (1, 2), (0, 1)]),
            ("cb,a,b,b,b->c", &[&[3, 4], &[10], &[4], &[4], &[4]], &[(2, 3), (2, 3), (0, 2), (0, 1)]),
            (
                "cba,ac,ca,abc,a->b",
                &[&[3, 4, 10], &[10, 3], &[3, 10], &[10, 4, 3], &[10]],
                &[(0, 3), (0, 1), (0, 2), (0, 1)],
            ),
            ("de,c,b,ad,dbce->bda", &[&[3, 2], &[2], &[4], &[3, 3], &[3, 4, 2, 2]], &[(0, 1), (2, 3), (0, 2), (0, 1)]),
            (
                "df,afcb,fe,fb,efd,ad->",
                &[&[3, 4], &[4, 4, 4, 3], &[4, 4], &[4, 3], &[4, 4, 3], &[4, 3]],
                &[(2, 4), (0, 4), (0, 3), (0, 2), (0, 1)],
            ),
            (
                "aed,bade,ea,b,be,cead->b",
                &[&[10, 3, 4], &[10, 10, 4, 3], &[3, 10], &[10], &[10, 3], &[3, 3, 10, 4]],
                &[(0, 2), (1, 2), (1, 2), (0, 1), (0, 1)],
            ),
            (
                "abcd,cd,bdca,abdc,dcba,cbd->b",
                &[&[10, 3, 4, 2], &[4, 2], &[3, 2, 4, 10], &[10, 3, 2, 4], &[2, 4, 3, 10], &[4, 3, 2]],
                &[(0, 2), (0, 3), (0, 1), (0, 2), (0, 1)],
            ),
            (
                "ba,cbd,b,a,da,cdb,acbd->",
                &[&[10, 10], &[10, 10, 10], &[10], &[10], &[10, 10], &[10, 10, 10], &[10, 10, 10, 10]],
                &[(0, 2), (0, 3), (0, 1), (0, 2), (0, 1), (0, 1)],
            ),
            (
                "adbc,abde,bad,ecb,ecb,be,cdbe->cd",
                &[&[3, 4, 3, 10], &[3, 3, 4, 2], &[3, 3, 4], &[2, 10, 3], &[2, 10, 3], &[3, 2], &[10, 4, 3, 2]],
                &[(3, 5), (3, 5), (3, 4), (0, 3), (0, 1), (0, 1)],
            ),
            (
                "abc,ac,bc,bca,b,b,cba->abc",
                &[&[2, 3, 3], &[2, 3], &[3, 3], &[3, 3, 2], &[3], &[3], &[3, 3, 2]],
                &[(0, 1), (1, 4), (1, 2), (1, 2), (0, 1), (0, 1)],
            ),
        ];
        let greedy: &[(&str, &[&[usize]], &[(usize, usize)])] = &[
            ("baf,dfab,a->df", &[&[2, 10, 10], &[50, 10, 10, 2], &[10]], &[(0, 1), (0, 1)]),
            ("cba,c,bac->", &[&[50, 10, 50], &[50], &[10, 50, 50]], &[(0, 2), (0, 1)]),
            ("hae,dbe,ch,h->", &[&[2, 3, 10], &[50, 50, 10], &[3, 2], &[2]], &[(0, 1), (1, 2), (0, 1)]),
            ("b,heig,ifh,fga->", &[&[50], &[2, 2, 10, 4], &[10, 3, 2], &[3, 4, 3]], &[(1, 2), (1, 2), (0, 1)]),
            ("c,cab,c,a,bc->ab", &[&[50], &[50, 10, 10], &[50], &[10], &[10, 50]], &[(0, 2), (0, 2), (1, 2), (0, 1)]),
            (
                "c,abc,ca,abc,abc->cab",
                &[&[10], &[3, 4, 10], &[10, 3], &[3, 4, 10], &[3, 4, 10]],
                &[(1, 3), (2, 3), (0, 1), (0, 1)],
            ),
            (
                "bac,bac,cb,c,bac,abc->a",
                &[&[4, 2, 2], &[4, 2, 2], &[2, 4], &[2], &[4, 2, 2], &[2, 4, 2]],
                &[(0, 1), (2, 4), (2, 3), (0, 2), (0, 1)],
            ),
            (
                "bacd,cd,dca,b,ca,b->bac",
                &[&[2, 50, 4, 50], &[4, 50], &[50, 4, 50], &[2], &[4, 50], &[2]],
                &[(3, 5), (0, 2), (0, 3), (0, 1), (0, 1)],
            ),
            (
                "badc,dbac,c,ad,a,cda,bdca->c",
                &[&[10, 2, 2, 3], &[2, 10, 2, 3], &[3], &[2, 2], &[2], &[3, 2, 2], &[10, 2, 3, 2]],
                &[(0, 1), (4, 5), (3, 4), (1, 3), (1, 2), (0, 1)],
            ),
            (
                "cbea,dc,ab,b,dab,dbc,cdbe->aec",
                &[&[3, 50, 4, 2], &[3, 3], &[2, 50], &[50], &[3, 2, 50], &[3, 50, 3], &[3, 3, 50, 4]],
                &[(5, 6), (0, 2), (1, 2), (0, 3), (0, 2), (0, 1)],
            ),
            (
                "bhca,hegf,hdec,dfh,gcbe,b,edfa,eh->gbc",
                &[
                    &[4, 3, 10, 4],
                    &[3, 50, 3, 2],
                    &[3, 4, 50, 10],
                    &[4, 2, 3],
                    &[3, 10, 4, 50],
                    &[4],
                    &[50, 4, 2, 4],
                    &[50, 3],
                ],
                &[(1, 7), (2, 6), (4, 5), (1, 4), (1, 3), (0, 2), (0, 1)],
            ),
            (
                "eb,eca,fe,bfc,fabe,ebad,a,f->fd",
                &[&[50, 10], &[50, 50, 10], &[4, 50], &[10, 4, 50], &[4, 10, 10, 50], &[50, 10, 10, 10], &[10], &[4]],
                &[(1, 3), (2, 6), (0, 5), (1, 4), (0, 3), (0, 2), (0, 1)],
            ),
            (
                "ae,acb,e,cd,deab,db,bcd,bea,edab->d",
                &[
                    &[2, 10],
                    &[2, 50, 50],
                    &[10],
                    &[50, 50],
                    &[50, 10, 2, 50],
                    &[50, 50],
                    &[50, 50, 50],
                    &[50, 10, 2],
                    &[10, 50, 2, 50],
                ],
                &[(4, 8), (3, 5), (1, 6), (4, 5), (2, 4), (2, 3), (0, 2), (0, 1)],
            ),
            (
                "dage,eg,efab,g,i,chfd,cai,ied,efi->bdf",
                &[
                    &[2, 3, 3, 50],
                    &[50, 3],
                    &[50, 4, 3, 2],
                    &[3],
                    &[3],
                    &[3, 2, 4, 2],
                    &[3, 3, 3],
                    &[3, 50, 2],
                    &[50, 4, 3],
                ],
                &[(0, 1), (1, 7), (1, 3), (1, 5), (1, 2), (2, 3), (1, 2), (0, 1)],
            ),
            (
                "bi,igab,afdi,d,c,g,cb,dfga,aeg,c->",
                &[
                    &[10, 2],
                    &[2, 10, 2, 10],
                    &[2, 3, 50, 2],
                    &[50],
                    &[4],
                    &[10],
                    &[4, 10],
                    &[50, 3, 10, 2],
                    &[2, 2, 10],
                    &[4],
                ],
                &[(4, 9), (2, 6), (2, 7), (1, 4), (4, 5), (1, 4), (0, 3), (0, 2), (0, 1)],
            ),
            (
                "efj,hbjf,hif,bgea,gia,cda,chbj,if,efb,c->i",
                &[
                    &[50, 50, 10],
                    &[50, 10, 10, 50],
                    &[50, 10, 50],
                    &[10, 2, 50, 4],
                    &[2, 10, 4],
                    &[3, 3, 4],
                    &[3, 50, 10, 10],
                    &[10, 50],
                    &[50, 50, 10],
                    &[3],
                ],
                &[(2, 7), (4, 7), (3, 7), (2, 6), (0, 3), (3, 4), (0, 1), (1, 2), (0, 1)],
            ),
            (
                "hjb,bcge,ehgc,bai,jg,fad,ihcg,g,i,ecdb,gc->",
                &[
                    &[50, 50, 50],
                    &[50, 50, 10, 2],
                    &[2, 50, 10, 50],
                    &[50, 50, 4],
                    &[50, 10],
                    &[2, 50, 4],
                    &[4, 50, 50, 10],
                    &[10],
                    &[4],
                    &[2, 50, 4, 50],
                    &[10, 50],
                ],
                &[(0, 4), (2, 3), (5, 8), (0, 5), (2, 4), (1, 2), (0, 4), (2, 3), (1, 2), (0, 1)],
            ),
            (
                "bc,cb,bc,ba,c,acb,acb,cb,cb,a,acb->ab",
                &[
                    &[3, 10],
                    &[10, 3],
                    &[3, 10],
                    &[3, 3],
                    &[10],
                    &[3, 10, 3],
                    &[3, 10, 3],
                    &[10, 3],
                    &[10, 3],
                    &[3],
                    &[3, 10, 3],
                ],
                &[(0, 1), (0, 9), (2, 3), (2, 6), (2, 6), (3, 4), (3, 4), (1, 3), (0, 2), (0, 1)],
            ),
            (
                "bcad,ba,cbad,acdb,cdab,cbda,d,acb,db,a,adcb,cb,d,bac,dbca,b->",
                &[
                    &[50, 4, 3, 10],
                    &[50, 3],
                    &[4, 50, 3, 10],
                    &[3, 4, 10, 50],
                    &[4, 10, 3, 50],
                    &[4, 50, 10, 3],
                    &[10],
                    &[3, 4, 50],
                    &[10, 50],
                    &[3],
                    &[3, 10, 4, 50],
                    &[4, 50],
                    &[10],
                    &[50, 3, 4],
                    &[10, 50, 4, 3],
                    &[50],
                ],
                &[
                    (0, 2),
                    (1, 14),
                    (1, 13),
                    (1, 12),
                    (5, 11),
                    (1, 6),
                    (1, 5),
                    (4, 6),
                    (6, 7),
                    (3, 6),
                    (1, 5),
                    (3, 4),
                    (0, 3),
                    (1, 2),
                    (0, 1),
                ],
            ),
            (
                "adc,a,acd,d,ad,ac,dcb,db,d,abcd,a,da,a,cadb,c,d->b",
                &[
                    &[2, 2, 50],
                    &[2],
                    &[2, 50, 2],
                    &[2],
                    &[2, 2],
                    &[2, 50],
                    &[2, 50, 4],
                    &[2, 4],
                    &[2],
                    &[2, 4, 50, 2],
                    &[2],
                    &[2, 2],
                    &[2],
                    &[50, 2, 2, 4],
                    &[50],
                    &[2],
                ],
                &[
                    (0, 2),
                    (1, 6),
                    (0, 6),
                    (0, 5),
                    (4, 10),
                    (3, 4),
                    (4, 6),
                    (1, 7),
                    (3, 7),
                    (0, 6),
                    (1, 5),
                    (0, 4),
                    (0, 3),
                    (0, 2),
                    (0, 1),
                ],
            ),
            (
                "geb,hfea,eagd,d,h,abdf,cad,bhfg,abdf,hb,fd,ad,hfe,h,h,b,dgeb,gch,ab,h->",
                &[
                    &[10, 3, 3],
                    &[4, 2, 3, 50],
                    &[3, 50, 10, 50],
                    &[50],
                    &[4],
                    &[50, 3, 50, 2],
                    &[2, 50, 50],
                    &[3, 4, 2, 10],
                    &[50, 3, 50, 2],
                    &[4, 3],
                    &[2, 50],
                    &[50, 50],
                    &[4, 2, 3],
                    &[4],
                    &[4],
                    &[3],
                    &[50, 10, 3, 3],
                    &[10, 2, 4],
                    &[50, 3],
                    &[4],
                ],
                &[
                    (5, 8),
                    (4, 11),
                    (10, 17),
                    (14, 16),
                    (2, 8),
                    (11, 12),
                    (6, 13),
                    (0, 8),
                    (1, 2),
                    (0, 3),
                    (0, 1),
                    (1, 2),
                    (0, 3),
                    (4, 5),
                    (1, 2),
                    (1, 3),
                    (0, 1),
                    (0, 2),
                    (0, 1),
                ],
            ),
            (
                "aehi,acbi,b,bhj,gfj,gaeh,gbha,fb,ce,cgjb,bdfj,ei,ei,je,dgha,ie,dgfb,he,g,cd->eh",
                &[
                    &[10, 3, 4, 50],
                    &[10, 4, 2, 50],
                    &[2],
                    &[2, 4, 3],
                    &[50, 10, 3],
                    &[50, 10, 3, 4],
                    &[50, 2, 4, 10],
                    &[10, 2],
                    &[4, 3],
                    &[4, 50, 3, 2],
                    &[2, 50, 10, 3],
                    &[3, 50],
                    &[3, 50],
                    &[3, 3],
                    &[50, 50, 4, 10],
                    &[50, 3],
                    &[50, 50, 10, 2],
                    &[4, 3],
                    &[50],
                    &[4, 50],
                ],
                &[
                    (11, 12),
                    (13, 18),
                    (0, 17),
                    (0, 16),
                    (2, 13),
                    (4, 7),
                    (4, 11),
                    (0, 1),
                    (3, 11),
                    (9, 10),
                    (0, 1),
                    (4, 6),
                    (5, 6),
                    (0, 6),
                    (0, 5),
                    (0, 3),
                    (2, 3),
                    (1, 2),
                    (0, 1),
                ],
            ),
            (
                "bd,c,da,abd,bca,dbac,cdba,a,acdb,dab,d,badc,c,bdc,b,bad,bdc,cbd,dbca,badc,cd,cda,c,c->d",
                &[
                    &[2, 50],
                    &[4],
                    &[50, 3],
                    &[3, 2, 50],
                    &[2, 4, 3],
                    &[50, 2, 3, 4],
                    &[4, 50, 2, 3],
                    &[3],
                    &[3, 4, 50, 2],
                    &[50, 3, 2],
                    &[50],
                    &[2, 3, 50, 4],
                    &[4],
                    &[2, 50, 4],
                    &[2],
                    &[2, 3, 50],
                    &[2, 50, 4],
                    &[4, 2, 50],
                    &[50, 2, 4, 3],
                    &[2, 3, 50, 4],
                    &[4, 50],
                    &[4, 50, 3],
                    &[4],
                    &[4],
                ],
                &[
                    (5, 6),
                    (6, 22),
                    (3, 6),
                    (6, 19),
                    (1, 6),
                    (7, 16),
                    (5, 7),
                    (6, 16),
                    (6, 12),
                    (6, 14),
                    (8, 10),
                    (8, 12),
                    (7, 10),
                    (8, 10),
                    (7, 9),
                    (6, 8),
                    (1, 7),
                    (0, 6),
                    (0, 5),
                    (3, 4),
                    (0, 3),
                    (1, 2),
                    (0, 1),
                ],
            ),
            (
                "bc,b,abc,acb,cb,ab,cba,cb,acb,c,a,bc,ab,ca,cab,bca,bca,abc,cba,cab,c,bc,bac,bc->a",
                &[
                    &[50, 4],
                    &[50],
                    &[10, 50, 4],
                    &[10, 4, 50],
                    &[4, 50],
                    &[10, 50],
                    &[4, 50, 10],
                    &[4, 50],
                    &[10, 4, 50],
                    &[4],
                    &[10],
                    &[50, 4],
                    &[10, 50],
                    &[4, 10],
                    &[4, 10, 50],
                    &[50, 4, 10],
                    &[50, 4, 10],
                    &[10, 50, 4],
                    &[4, 50, 10],
                    &[4, 10, 50],
                    &[4],
                    &[50, 4],
                    &[50, 10, 4],
                    &[50, 4],
                ],
                &[
                    (2, 3),
                    (0, 2),
                    (2, 20),
                    (2, 19),
                    (2, 18),
                    (4, 17),
                    (1, 4),
                    (4, 14),
                    (4, 15),
                    (4, 14),
                    (4, 13),
                    (4, 12),
                    (4, 11),
                    (1, 4),
                    (3, 6),
                    (3, 6),
                    (3, 6),
                    (3, 5),
                    (4, 5),
                    (0, 4),
                    (1, 3),
                    (1, 2),
                    (0, 1),
                ],
            ),
        ];
        for (subscripts, shapes, path) in auto {
            assert_eq!(planned_path(subscripts, shapes, EinsumPathStrategy::Auto), path.to_vec(), "{subscripts}");
        }
        for (subscripts, shapes, path) in optimal {
            assert_eq!(planned_path(subscripts, shapes, EinsumPathStrategy::Optimal), path.to_vec(), "{subscripts}");
        }
        for (subscripts, shapes, path) in greedy {
            assert_eq!(planned_path(subscripts, shapes, EinsumPathStrategy::Greedy), path.to_vec(), "{subscripts}");
        }
    }

    #[test]
    fn test_python_integer_set() {
        // Each case starts from `set(range(count))` and repeatedly replaces two values by a new one, like the path
        // searches do, recording `list(remaining)` after every replacement in CPython 3.13.
        let cases: &[(usize, &[(usize, usize, &[usize])])] = &[
            (3, &[(0, 1, &[2, 3]), (2, 3, &[4])]),
            (5, &[(0, 4, &[1, 2, 3, 5]), (3, 5, &[1, 2, 6]), (1, 6, &[2, 7]), (2, 7, &[8])]),
            (6, &[(3, 5, &[0, 1, 2, 4, 6]), (0, 1, &[2, 4, 6, 7]), (2, 6, &[8, 4, 7]), (4, 7, &[8, 9]), (8, 9, &[10])]),
            (
                8,
                &[
                    (0, 4, &[1, 2, 3, 5, 6, 7, 8]),
                    (5, 6, &[1, 2, 3, 7, 8, 9]),
                    (1, 2, &[3, 7, 8, 9, 10]),
                    (3, 7, &[8, 9, 10, 11]),
                    (9, 10, &[8, 11, 12]),
                    (8, 12, &[11, 13]),
                    (11, 13, &[14]),
                ],
            ),
            (
                9,
                &[
                    (1, 3, &[0, 2, 4, 5, 6, 7, 8, 9]),
                    (0, 7, &[2, 4, 5, 6, 8, 9, 10]),
                    (8, 9, &[2, 4, 5, 6, 10, 11]),
                    (2, 10, &[4, 5, 6, 11, 12]),
                    (4, 5, &[11, 12, 13, 6]),
                    (12, 13, &[11, 6, 14]),
                    (11, 14, &[6, 15]),
                    (6, 15, &[16]),
                ],
            ),
            (
                11,
                &[
                    (7, 9, &[0, 1, 2, 3, 4, 5, 6, 8, 10, 11]),
                    (4, 5, &[0, 1, 2, 3, 6, 8, 10, 11, 12]),
                    (2, 3, &[0, 1, 6, 8, 10, 11, 12, 13]),
                    (0, 8, &[1, 6, 10, 11, 12, 13, 14]),
                    (10, 12, &[1, 6, 11, 13, 14, 15]),
                    (13, 14, &[16, 1, 6, 11, 15]),
                    (11, 15, &[16, 1, 6, 17]),
                    (16, 17, &[1, 18, 6]),
                    (1, 18, &[19, 6]),
                    (6, 19, &[20]),
                ],
            ),
            (
                12,
                &[
                    (2, 5, &[0, 1, 3, 4, 6, 7, 8, 9, 10, 11, 12]),
                    (8, 9, &[0, 1, 3, 4, 6, 7, 10, 11, 12, 13]),
                    (0, 1, &[3, 4, 6, 7, 10, 11, 12, 13, 14]),
                    (11, 14, &[3, 4, 6, 7, 10, 12, 13, 15]),
                    (12, 15, &[16, 3, 4, 6, 7, 10, 13]),
                    (6, 10, &[16, 17, 3, 4, 7, 13]),
                    (13, 16, &[17, 18, 3, 4, 7]),
                    (3, 17, &[18, 19, 4, 7]),
                    (4, 7, &[18, 19, 20]),
                    (18, 19, &[20, 21]),
                    (20, 21, &[22]),
                ],
            ),
            (
                13,
                &[
                    (9, 10, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13]),
                    (7, 12, &[0, 1, 2, 3, 4, 5, 6, 8, 11, 13, 14]),
                    (4, 6, &[0, 1, 2, 3, 5, 8, 11, 13, 14, 15]),
                    (0, 8, &[1, 2, 3, 5, 11, 13, 14, 15, 16]),
                    (13, 15, &[16, 1, 2, 3, 17, 5, 11, 14]),
                    (3, 11, &[16, 1, 2, 17, 18, 5, 14]),
                    (1, 14, &[16, 17, 2, 18, 19, 5]),
                    (2, 5, &[16, 17, 18, 19, 20]),
                    (17, 18, &[16, 19, 20, 21]),
                    (19, 21, &[16, 20, 22]),
                    (20, 22, &[16, 23]),
                    (16, 23, &[24]),
                ],
            ),
            (
                14,
                &[
                    (6, 7, &[0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14]),
                    (4, 10, &[0, 1, 2, 3, 5, 8, 9, 11, 12, 13, 14, 15]),
                    (2, 9, &[0, 1, 3, 5, 8, 11, 12, 13, 14, 15, 16]),
                    (8, 14, &[0, 1, 3, 5, 11, 12, 13, 15, 16, 17]),
                    (12, 13, &[0, 1, 3, 5, 11, 15, 16, 17, 18]),
                    (5, 16, &[0, 1, 17, 3, 18, 19, 11, 15]),
                    (0, 3, &[1, 17, 18, 19, 20, 11, 15]),
                    (11, 20, &[1, 17, 18, 19, 21, 15]),
                    (15, 21, &[1, 18, 19, 17, 22]),
                    (1, 19, &[17, 18, 22, 23]),
                    (18, 23, &[24, 17, 22]),
                    (17, 22, &[24, 25]),
                    (24, 25, &[26]),
                ],
            ),
            (
                20,
                &[
                    (11, 17, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20]),
                    (10, 20, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 18, 19, 21]),
                    (4, 19, &[0, 1, 2, 3, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 18, 21, 22]),
                    (1, 18, &[0, 2, 3, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 21, 22, 23]),
                    (8, 16, &[0, 2, 3, 5, 6, 7, 9, 12, 13, 14, 15, 21, 22, 23, 24]),
                    (9, 24, &[0, 2, 3, 5, 6, 7, 12, 13, 14, 15, 21, 22, 23, 25]),
                    (2, 13, &[0, 3, 5, 6, 7, 12, 14, 15, 21, 22, 23, 25, 26]),
                    (14, 23, &[0, 3, 5, 6, 7, 12, 15, 21, 22, 25, 26, 27]),
                    (0, 6, &[3, 5, 7, 12, 15, 21, 22, 25, 26, 27, 28]),
                    (5, 12, &[3, 7, 15, 21, 22, 25, 26, 27, 28, 29]),
                    (15, 27, &[3, 7, 21, 22, 25, 26, 28, 29, 30]),
                    (7, 26, &[3, 21, 22, 25, 28, 29, 30, 31]),
                    (3, 31, &[32, 21, 22, 25, 28, 29, 30]),
                    (21, 29, &[32, 33, 22, 25, 28, 30]),
                    (25, 32, &[33, 34, 22, 28, 30]),
                    (22, 30, &[33, 34, 35, 28]),
                    (28, 35, &[33, 34, 36]),
                    (33, 34, &[36, 37]),
                    (36, 37, &[38]),
                ],
            ),
            (
                24,
                &[
                    (11, 19, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24]),
                    (12, 16, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 17, 18, 20, 21, 22, 23, 24, 25]),
                    (3, 18, &[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 17, 20, 21, 22, 23, 24, 25, 26]),
                    (20, 21, &[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 17, 22, 23, 24, 25, 26, 27]),
                    (10, 23, &[0, 1, 2, 4, 5, 6, 7, 8, 9, 13, 14, 15, 17, 22, 24, 25, 26, 27, 28]),
                    (2, 5, &[0, 1, 4, 6, 7, 8, 9, 13, 14, 15, 17, 22, 24, 25, 26, 27, 28, 29]),
                    (6, 17, &[0, 1, 4, 7, 8, 9, 13, 14, 15, 22, 24, 25, 26, 27, 28, 29, 30]),
                    (15, 29, &[0, 1, 4, 7, 8, 9, 13, 14, 22, 24, 25, 26, 27, 28, 30, 31]),
                    (9, 22, &[0, 1, 32, 4, 7, 8, 13, 14, 24, 25, 26, 27, 28, 30, 31]),
                    (0, 7, &[32, 1, 33, 4, 8, 13, 14, 24, 25, 26, 27, 28, 30, 31]),
                    (24, 27, &[32, 1, 33, 34, 4, 8, 13, 14, 25, 26, 28, 30, 31]),
                    (8, 33, &[32, 1, 34, 35, 4, 13, 14, 25, 26, 28, 30, 31]),
                    (1, 31, &[32, 34, 35, 4, 36, 13, 14, 25, 26, 28, 30]),
                    (26, 34, &[32, 35, 4, 36, 37, 13, 14, 25, 28, 30]),
                    (13, 28, &[32, 35, 4, 37, 36, 38, 14, 25, 30]),
                    (35, 38, &[32, 4, 37, 36, 30, 39, 25, 14]),
                    (25, 39, &[32, 4, 37, 36, 30, 40, 14]),
                    (14, 40, &[32, 4, 37, 36, 41, 30]),
                    (37, 41, &[32, 4, 36, 42, 30]),
                    (32, 42, &[43, 4, 36, 30]),
                    (30, 36, &[43, 4, 44]),
                    (4, 44, &[43, 45]),
                    (43, 45, &[46]),
                ],
            ),
        ];
        for (count, steps) in cases {
            let mut remaining = PythonIntegerSet::from_values(0..*count);
            assert_eq!(remaining.values(), (0..*count).collect::<Vec<_>>());
            for (index, (first, second, order)) in steps.iter().enumerate() {
                remaining = remaining
                    .difference(&PythonIntegerSet::from_values([*first, *second]))
                    .union(&PythonIntegerSet::from_values([count + index]));
                assert_eq!(remaining.len(), order.len());
                assert_eq!(remaining.values(), order.to_vec(), "count {count}, step {index}");
            }
        }
    }

    #[test]
    fn test_einsum_options_new() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap();
        let options = EinsumOptions::new();
        assert_eq!(options, EinsumOptions::default());
        assert_eq!(options.path(), &EinsumPathStrategy::Auto);
        assert_eq!(options.accumulation_type(), None);
        assert_eq!(options.precision(), DotPrecision::default());
        assert_eq!(options.output_sharding(), None);

        let options = EinsumOptions::new()
            .with_path(EinsumPathStrategy::Greedy)
            .with_accumulation_type(DataType::F32)
            .with_precision(DotAlgorithmPreset::BF16_BF16_F32)
            .with_output_sharding(sharding.clone());
        assert_eq!(options.path(), &EinsumPathStrategy::Greedy);
        assert_eq!(options.accumulation_type(), Some(DataType::F32));
        assert_eq!(options.precision(), DotPrecision::Preset(DotAlgorithmPreset::BF16_BF16_F32));
        assert_eq!(options.output_sharding(), Some(&sharding));
        assert_ne!(options, EinsumOptions::new());
    }

    #[test]
    fn test_einsum_plan_output_type() {
        let plan = |subscripts: &str, input_types: &[ArrayType], options: EinsumOptions| {
            subscripts.parse::<EinsumSpecification>().unwrap().plan(input_types, &options).unwrap()
        };
        let matrix =
            |data_type: DataType, rows: usize, columns: usize| ArrayType::new_static(data_type, [rows, columns]);

        // The output has the extents of its labels and the element type of the inputs.
        let types = [matrix(DataType::F32, 2, 3), matrix(DataType::F32, 3, 4), matrix(DataType::F32, 4, 5)];
        assert_eq!(plan("ij,jk,kl->li", &types, EinsumOptions::new()).output_type(), Ok(matrix(DataType::F32, 5, 2)));
        assert_eq!(plan("ij,jk,kl", &types, EinsumOptions::new()).output_type(), Ok(matrix(DataType::F32, 2, 5)),);
        assert_eq!(
            plan("ii->", &[matrix(DataType::I32, 3, 3)], EinsumOptions::new()).output_type(),
            Ok(ArrayType::scalar(DataType::I32)),
        );

        // An accumulation type is the element type of the output, including for an einsum without a contraction.
        let options = EinsumOptions::new().with_accumulation_type(DataType::F32);
        let narrow = [matrix(DataType::BF16, 2, 3), matrix(DataType::BF16, 3, 4), matrix(DataType::BF16, 4, 5)];
        assert_eq!(plan("ij,jk,kl->il", &narrow, options.clone()).output_type(), Ok(matrix(DataType::F32, 2, 5)));
        assert_eq!(plan("ij->ji", &narrow[..1], options).output_type(), Ok(matrix(DataType::F32, 3, 2)));

        // The inferred type is the type of the value that the plan produces.
        let inputs = [reference_input(0, &[2, 3]), reference_input(1, &[3, 4]), reference_input(2, &[4, 5])];
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let plan = plan("ij,jk,kl->li", &input_types, EinsumOptions::new());
        assert_eq!(plan.contract(&inputs).unwrap().r#type().into_owned(), plan.output_type().unwrap());

        // Invalid steps are reported without executing them.
        let options = EinsumOptions::new().with_precision(DotAlgorithmPreset::F32_F32_F32);
        let integers = [matrix(DataType::I32, 2, 3), matrix(DataType::I32, 3, 4)];
        let plan = "ij,jk".parse::<EinsumSpecification>().unwrap().plan(&integers, &options).unwrap();
        assert_eq!(
            plan.output_type(),
            Err(ProgramError::Type(TypeError::invalid(
                "`dot` algorithms require floating-point inputs but got `i32` and `i32`",
            ))),
        );
    }

    #[test]
    fn test_einsum_plan_contract() {
        // A plan can be executed any number of times, including on other inputs with the same ranks.
        let specification = "ij,jk->ik".parse::<EinsumSpecification>().unwrap();
        let inputs = [reference_input(0, &[2, 3]), reference_input(1, &[3, 2])];
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let plan = specification.plan(&input_types, &EinsumOptions::new()).unwrap();
        let expected = Array::matrix(2, 2, vec![8.0, -1.0, -15.0, 2.0]).unwrap();
        assert_eq!(plan.contract(&inputs), Ok(expected.clone()));
        assert_eq!(plan.contract(&inputs), Ok(expected));
        let others = [Array::matrix(1, 2, vec![1.0, 2.0]).unwrap(), Array::matrix(2, 1, vec![3.0, 4.0]).unwrap()];
        assert_eq!(plan.contract(&others), Ok(Array::matrix(1, 1, vec![11.0]).unwrap()));

        // The number of inputs and their ranks must match the plan.
        assert_eq!(
            plan.contract(&inputs[..1]),
            Err(ProgramError::Type(TypeError::invalid("`einsum` subscripts `ij,jk->ik` have 2 inputs but got 1"))),
        );
        assert_eq!(
            plan.contract(&[inputs[0].clone(), Array::vector(vec![1.0, 2.0, 3.0]).unwrap()]),
            Err(ProgramError::Type(TypeError::invalid(
                "`einsum` subscript 1 of `ij,jk->ik` does not match the rank 1 of its input",
            ))),
        );
    }

    #[test]
    fn test_einsum_plan_contract_staging() {
        let stage = |subscripts: &'static str, shapes: &[&[usize]], options: EinsumOptions, data_type: DataType| {
            let input_types =
                shapes.iter().map(|shape| ArrayType::new_static(data_type, shape.to_vec())).collect::<Vec<_>>();
            let (_, program) = ArrayContext::trace(
                move |inputs: Vec<DomainTracer<ArrayContext>>| {
                    DomainTracer::<ArrayContext>::einsum_with_options(&subscripts.parse()?, inputs.iter(), &options)
                },
                input_types,
            )
            .unwrap();
            program.to_flat_program().to_string()
        };

        // A matrix product is one dot of its inputs in their order.
        assert_eq!(
            stage("ij,jk->ik", &[&[2, 3], &[3, 4]], EinsumOptions::new(), DataType::F64),
            indoc! {"
                lambda %0:f64[2, 3], %1:f64[3, 4] .
                let %2:f64[2, 4] = dot [
                    dimensions=(lhs_contracting=[1], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                ] %0 %1
                in (%2)
            "}
            .trim_end(),
        );

        // Exchanging the inputs of the contraction produces the output order without a transpose, like in JAX.
        assert_eq!(
            stage("ijk,kl->ijl", &[&[2, 3, 4], &[4, 5]], EinsumOptions::new(), DataType::F64),
            indoc! {"
                lambda %0:f64[2, 3, 4], %1:f64[4, 5] .
                let %2:f64[2, 3, 5] = dot [
                    dimensions=(lhs_contracting=[2], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                ] %0 %1
                in (%2)
            "}
            .trim_end(),
        );
        assert_eq!(
            stage("ij,jk->ki", &[&[2, 3], &[3, 4]], EinsumOptions::new(), DataType::F64),
            indoc! {"
                lambda %0:f64[2, 3], %1:f64[3, 4] .
                let %2:f64[4, 2] = dot [
                    dimensions=(lhs_contracting=[0], rhs_contracting=[1], lhs_batching=[], rhs_batching=[]),
                ] %1 %0
                in (%2)
            "}
            .trim_end(),
        );

        // A transpose follows only when neither order of the inputs produces the output order.
        assert_eq!(
            stage("ijk,kl->jil", &[&[2, 3, 4], &[4, 5]], EinsumOptions::new(), DataType::F64),
            indoc! {"
                lambda %0:f64[2, 3, 4], %1:f64[4, 5] .
                let %2:f64[5, 2, 3] = dot [
                    dimensions=(lhs_contracting=[0], rhs_contracting=[2], lhs_batching=[], rhs_batching=[]),
                ] %1 %0
                    %3:f64[3, 2, 5] = transpose [permutation=[2, 1, 0]] %2
                in (%3)
            "}
            .trim_end(),
        );

        // Shared labels that the output keeps are batching dimensions, in the order of the output.
        assert_eq!(
            stage("...ij,...jk->...ik", &[&[5, 2, 3], &[5, 3, 4]], EinsumOptions::new(), DataType::F64),
            indoc! {"
                lambda %0:f64[5, 2, 3], %1:f64[5, 3, 4] .
                let %2:f64[5, 2, 4] = dot [
                    dimensions=(lhs_contracting=[2], rhs_contracting=[1], lhs_batching=[0], rhs_batching=[0]),
                ] %0 %1
                in (%2)
            "}
            .trim_end(),
        );

        // A chain contracts along its path. The first result orders its labels by their appearance in its inputs,
        // which are `kl,jk->lj` and then `lj,ij->il`, exactly like the path that JAX documents for this einsum.
        assert_eq!(
            stage("ij,jk,kl->il", &[&[2, 3], &[3, 4], &[4, 2]], EinsumOptions::new(), DataType::F64),
            indoc! {"
                lambda %0:f64[2, 3], %1:f64[3, 4], %2:f64[4, 2] .
                let %3:f64[2, 3] = dot [
                    dimensions=(lhs_contracting=[0], rhs_contracting=[1], lhs_batching=[], rhs_batching=[]),
                ] %2 %1
                    %4:f64[2, 2] = dot [
                        dimensions=(lhs_contracting=[1], rhs_contracting=[1], lhs_batching=[], rhs_batching=[]),
                    ] %0 %3
                in (%4)
            "}
            .trim_end(),
        );

        // Labels that only one input contains are summed out of it, a repeated label is replaced by its diagonal,
        // and an axis of extent one is squeezed away before it is broadcast.
        assert_eq!(
            stage("ij,k->i", &[&[2, 3], &[4]], EinsumOptions::new(), DataType::F64),
            indoc! {"
                lambda %0:f64[2, 3], %1:f64[4] .
                let %2:f64[] = reduce [kind=sum, axes=[0]] %1
                    %3:f64[2] = reduce [kind=sum, axes=[1]] %0
                    %4:f64[2] = dot [
                        dimensions=(lhs_contracting=[], rhs_contracting=[], lhs_batching=[], rhs_batching=[]),
                    ] %3 %2
                in (%4)
            "}
            .trim_end(),
        );
        assert_eq!(
            stage("ii->", &[&[3, 3]], EinsumOptions::new(), DataType::F64),
            indoc! {"
                lambda %0:f64[3, 3] .
                let %1:i64[3, 1] = iota [type=i64[3, 1], dimension=0]
                    %2:f64[3] = gather [
                        dimensions=(offset=[], collapsed_slice=[0], start_index_map=[0], batching=[(1, 0)]),
                        slice_sizes=[1, 1],
                    ] %0 %1
                    %3:f64[] = reduce [kind=sum, axes=[0]] %2
                in (%3)
            "}
            .trim_end(),
        );
        assert_eq!(
            stage("iij,j->i", &[&[2, 2, 3], &[3]], EinsumOptions::new(), DataType::F64),
            indoc! {"
                lambda %0:f64[2, 2, 3], %1:f64[3] .
                let %2:i64[2, 3, 1] = iota [type=i64[2, 3, 1], dimension=0]
                    %3:f64[2, 3] = gather [
                        dimensions=(offset=[], collapsed_slice=[0], start_index_map=[0], batching=[(1, 0), (2, 1)]),
                        slice_sizes=[1, 1, 1],
                    ] %0 %2
                    %4:f64[2] = dot [
                        dimensions=(lhs_contracting=[1], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                    ] %3 %1
                in (%4)
            "}
            .trim_end(),
        );
        assert_eq!(
            stage("ij,ij->ij", &[&[2, 3], &[1, 3]], EinsumOptions::new(), DataType::F64),
            indoc! {"
                lambda %0:f64[2, 3], %1:f64[1, 3] .
                let %2:f64[3] = reshape [shape=[3]] %1
                    %3:f64[3, 2] = dot [
                        dimensions=(lhs_contracting=[], rhs_contracting=[], lhs_batching=[0], rhs_batching=[1]),
                    ] %2 %0
                    %4:f64[2, 3] = transpose [permutation=[1, 0]] %3
                in (%4)
            "}
            .trim_end(),
        );

        // The precision and the accumulation type reach every contraction. An earlier result at the accumulation
        // type is contracted with an input after converting the input, and sums convert to a wider type first.
        let options = EinsumOptions::new()
            .with_precision(DotAlgorithmPreset::BF16_BF16_F32)
            .with_accumulation_type(DataType::F32);
        assert_eq!(
            stage("ij,jk,kl->il", &[&[2, 3], &[3, 4], &[4, 2]], options.clone(), DataType::BF16),
            indoc! {"
                lambda %0:bf16[2, 3], %1:bf16[3, 4], %2:bf16[4, 2] .
                let %3:f32[2, 3] = dot [
                    dimensions=(lhs_contracting=[0], rhs_contracting=[1], lhs_batching=[], rhs_batching=[]),
                    precision=bf16_bf16_f32,
                    accumulation_type=f32,
                ] %2 %1
                    %4:f32[2, 3] = convert_element_type [data_type=f32] %0
                    %5:f32[2, 2] = dot [
                        dimensions=(lhs_contracting=[1], rhs_contracting=[1], lhs_batching=[], rhs_batching=[]),
                        precision=bf16_bf16_f32,
                        accumulation_type=f32,
                    ] %4 %3
                in (%5)
            "}
            .trim_end(),
        );
        assert_eq!(
            stage("ij->i", &[&[2, 3]], options.clone(), DataType::BF16),
            indoc! {"
                lambda %0:bf16[2, 3] .
                let %1:f32[2, 3] = convert_element_type [data_type=f32] %0
                    %2:f32[2] = reduce [kind=sum, axes=[1]] %1
                in (%2)
            "}
            .trim_end(),
        );
        assert_eq!(
            stage("ij->ji", &[&[2, 3]], options, DataType::BF16),
            indoc! {"
                lambda %0:bf16[2, 3] .
                let %1:bf16[3, 2] = transpose [permutation=[1, 0]] %0
                    %2:f32[3, 2] = convert_element_type [data_type=f32] %1
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_einsum_plan_contract_output_sharding() {
        // The last contraction receives the requested output sharding, in the order of its own axes when a
        // transpose follows it.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let sharding = |dimensions: Vec<ShardingDimension>| Sharding::new(mesh.clone(), dimensions).unwrap();
        let output_sharding = sharding(vec![
            ShardingDimension::sharded(["x"]),
            ShardingDimension::Replicated,
            ShardingDimension::sharded(["y"]),
        ]);
        let options = EinsumOptions::new().with_output_sharding(output_sharding.clone());
        let input_types =
            vec![ArrayType::new_static(DataType::F64, [2, 4, 6]), ArrayType::new_static(DataType::F64, [6, 4])];
        let specification = "ijk,kl->jil".parse::<EinsumSpecification>().unwrap();
        let plan = specification.plan(&input_types, &options).unwrap();
        assert_eq!(
            plan.output_type(),
            Ok(ArrayType::new_static(DataType::F64, [4, 2, 4]).with_sharding(output_sharding.clone()).unwrap()),
        );
        let (_, program) =
            ArrayContext::trace(|inputs: Vec<DomainTracer<ArrayContext>>| plan.contract(&inputs), input_types).unwrap();
        let dot_sharding = sharding(vec![
            ShardingDimension::sharded(["y"]),
            ShardingDimension::Replicated,
            ShardingDimension::sharded(["x"]),
        ]);
        assert_eq!(
            program.to_flat_program().to_string(),
            formatdoc! {"
                lambda %0:f64[2, 4, 6], %1:f64[6, 4] .
                let %2:f64[4, 2, 4][sharding={dot_sharding}] = dot [
                    dimensions=(lhs_contracting=[0], rhs_contracting=[2], lhs_batching=[], rhs_batching=[]),
                    output_sharding={dot_sharding},
                ] %1 %0
                    %3:f64[4, 2, 4][sharding={output_sharding}] = transpose [permutation=[2, 1, 0]] %2
                in (%3)
            "}
            .trim_end(),
        );

        // An einsum of one input has no contraction to place.
        let plan = "ij->ji"
            .parse::<EinsumSpecification>()
            .unwrap()
            .plan(
                &[ArrayType::new_static(DataType::F64, [2, 4])],
                &EinsumOptions::new().with_output_sharding(sharding(vec![ShardingDimension::Replicated; 2])),
            )
            .unwrap();
        assert_eq!(
            plan.output_type(),
            Err(ProgramError::Type(TypeError::invalid(
                "`einsum` output sharding requires a pairwise contraction as the last step, which an einsum of one \
                 input does not have",
            ))),
        );
    }

    #[test]
    fn test_einsum_plan_contract_with() {
        // The custom contraction receives every pairwise contraction of the plan with its dimension numbers.
        let specification = "ij,jk,kl->il".parse::<EinsumSpecification>().unwrap();
        let inputs = [reference_input(0, &[2, 3]), reference_input(1, &[3, 4]), reference_input(2, &[4, 2])];
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let plan = specification.plan(&input_types, &EinsumOptions::new()).unwrap();
        let mut contractions = Vec::new();
        let output = plan
            .contract_with(&inputs, |lhs: &Array, rhs: &Array, dimensions, output_sharding| {
                contractions.push((dimensions.clone(), output_sharding.cloned()));
                lhs.dot(rhs, dimensions)
            })
            .unwrap();
        assert_eq!(output, plan.contract(&inputs).unwrap());
        assert_eq!(
            contractions,
            vec![
                (DotDimensionNumbers::new(vec![0], vec![1], vec![], vec![]), None),
                (DotDimensionNumbers::new(vec![1], vec![1], vec![], vec![]), None),
            ],
        );

        // A contraction that quantizes its inputs first changes the result of every pairwise step. Here each input
        // of a contraction is rounded to `bf16`, whose eight significand bits round `257` to `256`.
        let lhs = Array::matrix(1, 2, vec![257.0, 1.0]).unwrap();
        let rhs = Array::matrix(2, 1, vec![1.0, 257.0]).unwrap();
        let input_types = vec![lhs.r#type().into_owned(), rhs.r#type().into_owned()];
        let plan = "ij,jk->ik"
            .parse::<EinsumSpecification>()
            .unwrap()
            .plan(&input_types, &EinsumOptions::new())
            .unwrap();
        let quantized = plan.contract_with(&[lhs.clone(), rhs.clone()], |lhs: &Array, rhs: &Array, dimensions, _| {
            let lhs = lhs.convert_element_type(DataType::BF16)?;
            let rhs = rhs.convert_element_type(DataType::BF16)?;
            lhs.dot_with_accumulation_type(&rhs, dimensions, DataType::F64)
        });
        assert_eq!(plan.contract(&[lhs, rhs]), Ok(Array::matrix(1, 1, vec![514.0]).unwrap()));
        assert_eq!(quantized, Ok(Array::matrix(1, 1, vec![512.0]).unwrap()));
    }

    #[test]
    fn test_einsum() {
        // These results were recorded from `numpy.einsum` for the inputs that `reference_input` creates. The first
        // table covers every kind of step, and the second one is a seeded random corpus of one to nine inputs with
        // repeated labels and implicit outputs.
        let picked: &[(&str, &[&[usize]], &[usize], &[f64])] = &[
            ("ij,jk->ik", &[&[2, 3], &[3, 2]], &[2, 2], &[8.0, -1.0, -15.0, 2.0]),
            ("ij,jk", &[&[2, 3], &[3, 2]], &[2, 2], &[8.0, -1.0, -15.0, 2.0]),
            ("i,i->", &[&[3], &[3]], &[], &[12.0]),
            ("i,j->ij", &[&[2], &[3]], &[2, 3], &[9.0, 0.0, -9.0, 3.0, 0.0, -3.0]),
            ("ii->i", &[&[3, 3]], &[3], &[-3.0, -2.0, -1.0]),
            ("ii", &[&[3, 3]], &[], &[-6.0]),
            ("ij->ji", &[&[2, 3]], &[3, 2], &[-3.0, 3.0, -1.0, -2.0, 1.0, 0.0]),
            ("ij->", &[&[2, 3]], &[], &[-2.0]),
            ("ij->j", &[&[2, 3]], &[3], &[0.0, -3.0, 1.0]),
            (
                "ijk->kji",
                &[&[2, 3, 2]],
                &[2, 3, 2],
                &[-3.0, 2.0, 1.0, -1.0, -2.0, 3.0, -1.0, -3.0, 3.0, 1.0, 0.0, -2.0],
            ),
            (
                "...ij,...jk->...ik",
                &[&[2, 2, 3], &[2, 3, 2]],
                &[2, 2, 2],
                &[8.0, -1.0, -15.0, 2.0, 3.0, -17.0, 3.0, 2.0],
            ),
            ("...ij,jk", &[&[2, 2, 3], &[3, 2]], &[2, 2, 2], &[8.0, -1.0, -15.0, 2.0, -17.0, 5.0, 2.0, 1.0]),
            ("...,...->...", &[&[2, 3], &[3]], &[2, 3], &[9.0, 0.0, 3.0, -9.0, 0.0, 0.0]),
            ("bij,bjk->bik", &[&[2, 2, 3], &[2, 3, 2]], &[2, 2, 2], &[8.0, -1.0, -15.0, 2.0, 3.0, -17.0, 3.0, 2.0]),
            ("ij,jk,kl->il", &[&[2, 3], &[3, 4], &[4, 2]], &[2, 2], &[-20.0, -40.0, 24.0, 7.0]),
            ("ijk,kl->ijl", &[&[2, 2, 3], &[3, 2]], &[2, 2, 2], &[8.0, -1.0, -15.0, 2.0, -17.0, 5.0, 2.0, 1.0]),
            ("ii,ij->j", &[&[2, 2], &[2, 3]], &[3], &[6.0, 6.0, -15.0]),
            ("iij->ij", &[&[2, 2, 3]], &[2, 3], &[-3.0, -1.0, 1.0, 1.0, 3.0, -2.0]),
            ("iij->j", &[&[2, 2, 3]], &[3], &[-2.0, 2.0, -1.0]),
            ("iji->j", &[&[2, 3, 2]], &[3], &[-6.0, 2.0, -4.0]),
            ("iii->i", &[&[2, 2, 2]], &[2], &[-3.0, -3.0]),
            ("iii", &[&[2, 2, 2]], &[], &[-6.0]),
            ("ij,ij->ij", &[&[2, 3], &[1, 3]], &[2, 3], &[9.0, 0.0, 3.0, -9.0, 0.0, 0.0]),
            ("ij,ij->i", &[&[2, 3], &[1, 3]], &[2], &[12.0, -9.0]),
            ("ij,jk->ik", &[&[2, 1], &[3, 2]], &[2, 2], &[-6.0, 9.0, -2.0, 3.0]),
            ("i,ij,j->", &[&[2], &[2, 3], &[3]], &[], &[-18.0]),
            (
                "ba,Ab",
                &[&[2, 3], &[4, 2]],
                &[4, 3],
                &[9.0, 3.0, -3.0, -12.0, -1.0, 3.0, -12.0, 2.0, 2.0, -12.0, 5.0, 1.0],
            ),
            ("ij,ji", &[&[2, 3], &[3, 2]], &[], &[10.0]),
            ("i,i,i->i", &[&[3], &[3], &[3]], &[3], &[-27.0, 0.0, -6.0]),
            (",i->i", &[&[], &[3]], &[3], &[9.0, 0.0, -9.0]),
            ("ab,cd->abcd", &[&[2, 2], &[1, 2]], &[2, 2, 1, 2], &[9.0, 0.0, 3.0, 0.0, -3.0, 0.0, -9.0, 0.0]),
            ("ab,bc,cd,de->ae", &[&[2, 3], &[3, 2], &[2, 3], &[3, 2]], &[2, 2], &[21.0, -89.0, -39.0, 168.0]),
            ("ab,ab,bc->c", &[&[2, 3], &[2, 3], &[3, 2]], &[2], &[-13.0, 7.0]),
            ("abc,cd,db->a", &[&[2, 3, 2], &[2, 3], &[3, 3]], &[2], &[-55.0, 27.0]),
        ];
        let random: &[(&str, &[&[usize]], &[usize], &[f64])] = &[
            ("bcd", &[&[2, 3, 2]], &[2, 3, 2], &[-3.0, -1.0, 1.0, 3.0, -2.0, 0.0, 2.0, -3.0, -1.0, 1.0, 3.0, -2.0]),
            ("dcb->cb", &[&[3, 2, 2]], &[2, 2], &[-6.0, 0.0, 6.0, -2.0]),
            ("b->b", &[&[2]], &[2], &[-3.0, -1.0]),
            ("dbeb->e", &[&[3, 2, 2, 2]], &[2], &[-4.0, -1.0]),
            (
                "bed,c->bed",
                &[&[3, 3, 2], &[3]],
                &[3, 3, 2],
                &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ),
            (
                "ad,aba->dab",
                &[&[2, 3], &[2, 2, 2]],
                &[3, 2, 2],
                &[9.0, -9.0, -6.0, -9.0, 3.0, -3.0, 4.0, 6.0, -3.0, 3.0, 0.0, 0.0],
            ),
            ("aa,e", &[&[2, 2], &[3]], &[3], &[0.0, 0.0, 0.0]),
            ("ccb,dee->c", &[&[3, 3, 2], &[3, 2, 2]], &[3], &[12.0, 0.0, -12.0]),
            ("ee,b->", &[&[3, 3], &[2]], &[], &[18.0]),
            ("cc,dca->", &[&[2, 2], &[3, 2, 2]], &[], &[3.0]),
            ("d,dcb,a->", &[&[2], &[2, 3, 2], &[2]], &[], &[-2.0]),
            (
                "bc,cde,acb->ecd",
                &[&[3, 3], &[3, 2, 2], &[3, 3, 3]],
                &[2, 3, 2],
                &[-90.0, 90.0, 8.0, 4.0, 0.0, 3.0, 0.0, -30.0, -8.0, -12.0, -9.0, -6.0],
            ),
            (
                "cdaa,ad,ac->dc",
                &[&[3, 3, 3, 3], &[3, 3], &[3, 3]],
                &[3, 3],
                &[-23.0, 6.0, -1.0, 8.0, 9.0, -9.0, 22.0, 3.0, -18.0],
            ),
            (
                "a,dc,ae->acd",
                &[&[3], &[3, 2], &[3, 3]],
                &[3, 2, 3],
                &[
                    -36.0, 36.0, 24.0, 0.0, -12.0, -24.0, 12.0, -12.0, -8.0, 0.0, 4.0, 8.0, 6.0, -6.0, -4.0, 0.0, 2.0,
                    4.0,
                ],
            ),
            (
                "edb,ee,d->bde",
                &[&[3, 3, 2], &[3, 3], &[3]],
                &[2, 3, 3],
                &[
                    -27.0, -12.0, 0.0, -3.0, -2.0, 0.0, -12.0, -12.0, 0.0, -9.0, 18.0, 0.0, -9.0, 2.0, 0.0, 0.0, 8.0,
                    0.0,
                ],
            ),
            ("eddc,bd,ace", &[&[2, 3, 3, 3], &[2, 3], &[3, 3, 2]], &[3, 2], &[-21.0, 11.0, -33.0, 16.0, 81.0, -7.0]),
            ("abcc,dcee,abb,be", &[&[3, 3, 2, 2], &[2, 2, 2, 2], &[3, 3, 3], &[3, 2]], &[2], &[187.0, 22.0]),
            (
                "cda,ac,ddea,d->ecd",
                &[&[2, 3, 2], &[2, 2], &[3, 3, 3, 2], &[3]],
                &[3, 2, 3],
                &[
                    90.0, -30.0, 0.0, -9.0, 2.0, 0.0, 72.0, -18.0, 0.0, -18.0, 0.0, 0.0, 54.0, 36.0, 0.0, -27.0, -2.0,
                    0.0,
                ],
            ),
            ("adae,cab,ed,cd->e", &[&[2, 3, 2, 2], &[3, 2, 3], &[2, 3], &[3, 3]], &[2], &[-15.0, -8.0]),
            ("dce,db,ddc,bc->e", &[&[3, 3, 3], &[3, 2], &[3, 3, 3], &[2, 3]], &[3], &[23.0, -11.0, 18.0]),
            (
                "b,aabd,cb,be,aa->dec",
                &[&[2], &[2, 2, 2, 3], &[3, 2], &[2, 3], &[2, 2]],
                &[3, 3, 3],
                &[
                    -297.0, -198.0, -99.0, 198.0, 132.0, 66.0, 0.0, 0.0, 0.0, 42.0, 12.0, -18.0, -18.0, 12.0, 42.0,
                    6.0, 12.0, 18.0, 234.0, 180.0, 126.0, -171.0, -150.0, -129.0, -9.0, -18.0, -27.0,
                ],
            ),
            (
                "ee,caa,d,db,ee->dba",
                &[&[3, 3], &[2, 3, 3], &[2], &[2, 2], &[3, 3]],
                &[2, 2, 3],
                &[0.0, 162.0, -54.0, 0.0, -108.0, 36.0, 0.0, 0.0, 0.0, 0.0, -36.0, 12.0],
            ),
            (
                "dd,cad,adb,bd,deb->bac",
                &[&[3, 3], &[2, 2, 3], &[2, 3, 2], &[2, 3], &[3, 3, 2]],
                &[2, 2, 2],
                &[-81.0, -21.0, -16.0, 8.0, -72.0, -18.0, 26.0, -146.0],
            ),
            (
                "e,dd,dd,ca,bc->ace",
                &[&[3], &[3, 3], &[3, 3], &[2, 2], &[3, 2]],
                &[2, 2, 3],
                &[-63.0, -21.0, 21.0, 0.0, 0.0, 0.0, 42.0, 14.0, -14.0, 126.0, 42.0, -42.0],
            ),
            (
                "bdbc,ab,aba,deec,cab,dd",
                &[&[2, 2, 2, 3], &[2, 2], &[2, 2, 2], &[2, 2, 2, 3], &[3, 2, 2], &[2, 2]],
                &[],
                &[-783.0],
            ),
            ("ba,a,aeb,d,dca,bca->", &[&[2, 3], &[3], &[3, 2, 2], &[3], &[3, 3, 3], &[2, 3, 3]], &[], &[4752.0]),
            (
                "c,ed,e,cea,e,cde,adb->ea",
                &[&[2], &[3, 2], &[3], &[2, 3, 3], &[3], &[2, 2, 3], &[3, 2, 3]],
                &[3, 3],
                &[-4374.0, 648.0, -486.0, 945.0, -432.0, 0.0, -216.0, -432.0, -480.0],
            ),
            (
                "aee,adae,a,cdd,d,bac,eb->d",
                &[&[3, 2, 2], &[3, 3, 3, 2], &[3], &[2, 3, 3], &[3], &[2, 3, 2], &[2, 2]],
                &[3],
                &[-4104.0, 1818.0, 0.0],
            ),
            (
                "b,ddc,cea,be,b,b,ae,ac,adba->ebd",
                &[&[2], &[3, 3, 3], &[3, 3, 3], &[2, 3], &[2], &[2], &[3, 3], &[3, 3], &[3, 3, 2, 3]],
                &[3, 2, 3],
                &[
                    -24786.0, 11178.0, -1458.0, 972.0, -918.0, -1080.0, 0.0, -3078.0, -1134.0, 1458.0, -4860.0,
                    -1215.0, 0.0, 0.0, 0.0, 324.0, -1035.0, 594.0,
                ],
            ),
            (
                "bce,e,beea,e,bb,c,aa,da,bae->d",
                &[&[2, 2, 3], &[3], &[2, 3, 3, 3], &[3], &[2, 2], &[2], &[3, 3], &[2, 3], &[2, 3, 3]],
                &[2],
                &[486.0, 972.0],
            ),
        ];
        for (subscripts, shapes, shape, values) in picked.iter().chain(random) {
            let inputs =
                shapes.iter().enumerate().map(|(index, shape)| reference_input(index, shape)).collect::<Vec<_>>();
            let expected = Array::from_elements(ArrayType::new_static(DataType::F64, shape.to_vec()), values).unwrap();
            assert_eq!(Array::einsum(subscripts, inputs.iter()), Ok(expected), "{subscripts}");
        }

        // Invalid subscripts and inputs are rejected.
        let matrix = reference_input(0, &[2, 3]);
        assert_eq!(
            Array::einsum("ij->jj", [&matrix]),
            Err(ProgramError::Type(TypeError::invalid("`einsum` output label `j` appears more than once"))),
        );
        assert_eq!(
            Array::einsum("ij,jk", [&matrix]),
            Err(ProgramError::Type(TypeError::invalid("`einsum` subscripts `ij,jk` have 2 inputs but got 1"))),
        );
    }

    #[test]
    fn test_einsum_with() {
        let matrix = Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let vector = Array::vector(vec![10.0, 100.0]).unwrap();
        assert_eq!(matrix.einsum_with("ij,j->i", [&vector]), Ok(Array::vector(vec![210.0, 430.0]).unwrap()));
        assert_eq!(matrix.einsum_with("ii", []), Ok(Array::scalar(5.0).unwrap()));
        assert_eq!(matrix.einsum_with("ij,j->i", [&vector]), Array::einsum("ij,j->i", [&matrix, &vector]));
    }

    #[test]
    fn test_einsum_with_options() {
        let specification = "ij,jk,kl->il".parse::<EinsumSpecification>().unwrap();
        let inputs = [reference_input(0, &[2, 3]), reference_input(1, &[3, 4]), reference_input(2, &[4, 2])];
        let expected = Array::einsum("ij,jk,kl->il", inputs.iter()).unwrap();

        // Every path strategy computes the same result.
        for path in [
            EinsumPathStrategy::Auto,
            EinsumPathStrategy::Optimal,
            EinsumPathStrategy::Greedy,
            EinsumPathStrategy::Unoptimized,
            EinsumPathStrategy::Explicit(vec![(0, 2), (0, 1)]),
        ] {
            let options = EinsumOptions::new().with_path(path.clone());
            assert_eq!(
                Array::einsum_with_options(&specification, inputs.iter(), &options),
                Ok(expected.clone()),
                "{path:?}",
            );
        }

        // An accumulation type is the element type of the output. The `bf16` inputs below are exact, and their
        // product `257 · 257 = 66049` is exact at `f32` but rounds to `66048` at `bf16`.
        let narrow = Array::matrix(1, 1, vec![bf16::from_f32(257.0)]).unwrap();
        assert_eq!(narrow.elements::<bf16>().unwrap(), vec![bf16::from_f32(256.0)]);
        let narrow = Array::matrix(1, 2, vec![bf16::from_f32(3.0), bf16::from_f32(0.5)]).unwrap();
        let options = EinsumOptions::new().with_accumulation_type(DataType::F32);
        assert_eq!(
            Array::einsum_with_options(&"ij,kj->ik".parse().unwrap(), [&narrow, &narrow], &options),
            Ok(Array::matrix(1, 1, vec![9.25f32]).unwrap()),
        );
        assert_eq!(
            Array::einsum_with_options(&"ij->i".parse().unwrap(), [&narrow], &options),
            Ok(Array::vector(vec![3.5f32]).unwrap()),
        );

        // Integer inputs accumulate at a floating-point type in every form of an einsum. A contraction converts its
        // inputs first, because a `dot` accumulates integers only at integer types, and a sum converts its input.
        let integers = Array::matrix(2, 2, vec![1i32, 2, 3, 4]).unwrap();
        let options = EinsumOptions::new().with_accumulation_type(DataType::F32);
        assert_eq!(
            Array::einsum_with_options(&"ij,jk->ik".parse().unwrap(), [&integers, &integers], &options),
            Ok(Array::matrix(2, 2, vec![7.0f32, 10.0, 15.0, 22.0]).unwrap()),
        );
        assert_eq!(
            Array::einsum_with_options(&"ij,jk->i".parse().unwrap(), [&integers, &integers], &options),
            Ok(Array::vector(vec![17.0f32, 37.0]).unwrap()),
        );
        assert_eq!(
            Array::einsum_with_options(&"ij->i".parse().unwrap(), [&integers], &options),
            Ok(Array::vector(vec![3.0f32, 7.0]).unwrap()),
        );
        let wide = EinsumOptions::new().with_accumulation_type(DataType::I64);
        assert_eq!(
            Array::einsum_with_options(&"ij,jk->ik".parse().unwrap(), [&integers, &integers], &wide),
            Ok(Array::matrix(2, 2, vec![7i64, 10, 15, 22]).unwrap()),
        );

        // The 8-bit floating-point types promote to no other type, and their sums still accumulate at a wider
        // accumulation type. The sum `7 + 7 + 7 = 21` below is exact at `f32` and is not representable at
        // `f8e4m3fn`, which holds four significant bits.
        let sevens = Array::matrix(1, 3, vec![7.0f32; 3]).unwrap().convert_element_type(DataType::F8E4M3FN).unwrap();
        let ones = Array::vector(vec![1.0f32; 2]).unwrap().convert_element_type(DataType::F8E4M3FN).unwrap();
        assert_eq!(
            Array::einsum_with_options(&"ij,k->i".parse().unwrap(), [&sevens, &ones], &options),
            Ok(Array::vector(vec![42.0f32]).unwrap()),
        );

        // A sum converts its input unless the input already has the type that the two promote to, like in JAX. The
        // types `bf16` and `f16` promote to `f32`, so the sum below runs at `f16`, where `256 + 1` is exact, while
        // `bf16` rounds it to `256`. An `i8` and a `u16` promote to `i32`, so the sum of two `100`s runs at `u16` and
        // does not wrap around.
        let narrow = Array::vector(vec![bf16::from_f32(256.0), bf16::from_f32(1.0)]).unwrap();
        let options = EinsumOptions::new().with_accumulation_type(DataType::F16);
        assert_eq!(
            Array::einsum_with_options(&"i->".parse().unwrap(), [&narrow], &options),
            Ok(Array::scalar(f16::from_f32(257.0)).unwrap()),
        );
        let bytes = Array::vector(vec![100i8, 100]).unwrap();
        let options = EinsumOptions::new().with_accumulation_type(DataType::U16);
        assert_eq!(
            Array::einsum_with_options(&"i->".parse().unwrap(), [&bytes], &options),
            Ok(Array::scalar(200u16).unwrap()),
        );

        // A contraction converts integers to a type of the other signedness that is at least as wide, which a `dot`
        // does not accumulate at itself, and a narrower type is rejected.
        let bytes = Array::vector(vec![1u8, 2]).unwrap();
        let options = EinsumOptions::new().with_accumulation_type(DataType::I32);
        assert_eq!(
            Array::einsum_with_options(&"i,i->".parse().unwrap(), [&bytes, &bytes], &options),
            Ok(Array::scalar(5i32).unwrap()),
        );
        let integers = Array::vector(vec![1i32, 2]).unwrap();
        let options = EinsumOptions::new().with_accumulation_type(DataType::I8);
        assert_eq!(
            Array::einsum_with_options(&"i,i->".parse().unwrap(), [&integers, &integers], &options),
            Err(ProgramError::Type(TypeError::invalid(
                "`dot` input data type `i32` cannot accumulate at data type `i8`",
            ))),
        );

        // Distinct 8-bit floating-point types are contracted natively when an accumulation type is requested.
        let matrix = Array::matrix(2, 2, vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let lhs = matrix.convert_element_type(DataType::F8E4M3FN).unwrap();
        let rhs = matrix.convert_element_type(DataType::F8E5M2).unwrap();
        let options = EinsumOptions::new().with_accumulation_type(DataType::F32);
        assert_eq!(
            Array::einsum_with_options(&"ij,jk->ik".parse().unwrap(), [&lhs, &rhs], &options),
            Ok(Array::matrix(2, 2, vec![7.0f32, 10.0, 15.0, 22.0]).unwrap()),
        );
        assert_eq!(
            Array::einsum_with_options(
                &"ij,jk,k->i".parse().unwrap(),
                [&lhs, &rhs, &lhs.einsum_with("ij->j", [])?],
                &options
            ),
            Array::einsum("ij,jk,k->i", [&matrix, &matrix, &matrix.einsum_with("ij->j", []).unwrap()]),
        );
        assert_eq!(
            Array::einsum("ij,jk->ik", [&lhs, &rhs]),
            Err(ProgramError::Type(TypeError::invalid(
                "`einsum` inputs must have the same element type but got `f8e4m3fn` and `f8e5m2`",
            ))),
        );

        // A precision that the inputs do not support is rejected by the contraction.
        let integers = Array::matrix(1, 1, vec![2i32]).unwrap();
        let options = EinsumOptions::new().with_precision(DotAlgorithmPreset::F32_F32_F32);
        assert_eq!(
            Array::einsum_with_options(&"ij,jk".parse().unwrap(), [&integers, &integers], &options),
            Err(ProgramError::Type(TypeError::invalid(
                "`dot` algorithms require floating-point inputs but got `i32` and `i32`",
            ))),
        );
    }

    #[test]
    fn test_einsum_provenance() {
        // The instructions of an inline einsum are attributed to a scope named after its specification, and so are
        // the ones of a composite call that contracts inline.
        let scope = |subscripts: &str| Provenance::scope(ProvenanceScope::new(subscripts), Provenance::unknown());
        let function = einsum::<ArrayContext>("ij , jk -> ik").unwrap();
        let (_, program) = ArrayContext::trace(
            |inputs: Vec<DomainTracer<ArrayContext>>| {
                let chain = DomainTracer::<ArrayContext>::einsum("ij,jk,kl->il", inputs.iter())?;
                function.call(vec![chain, inputs[1].clone()])
            },
            vec![ArrayType::new_static(DataType::F64, [2, 2]); 3],
        )
        .unwrap();
        let provenances = program
            .instructions()
            .iter()
            .map(|instruction| instruction.provenance().clone())
            .collect::<Vec<_>>();
        assert_eq!(provenances, vec![scope("ij,jk,kl->il"), scope("ij,jk,kl->il"), scope("ij,jk->ik")]);
    }

    #[test]
    fn test_einsum_manual_variation() {
        // The pairwise contraction aligns the manual variation of its inputs, and the inferred output type of the
        // plan models that alignment.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let varying = Sharding::replicated(mesh.clone(), 2).with_varying_manual_axes(["x"]).unwrap();
        let lhs_type = ArrayType::new_static(DataType::F64, [2, 3]).with_sharding(varying.clone()).unwrap();
        let rhs_type = ArrayType::new_static(DataType::F64, [3, 4]);
        let output_type = ArrayType::new_static(DataType::F64, [2, 4]).with_sharding(varying).unwrap();
        let (outputs, program) = DomainTracingContext::<ArrayContext>::trace_with_named_axes(
            |inputs| Ok(vec![Einsum::einsum("ij,jk->ik", inputs.iter())?]),
            vec![lhs_type.clone(), rhs_type.clone()],
            vec![("x".to_string(), NamedAxis::Mesh { mesh, axis: 0, size: 2, checks_manual_variation: true })],
        )
        .unwrap();
        assert_eq!(outputs, vec![output_type.clone()]);
        assert_eq!(program.to_string(), "");
        let plan = "ij,jk->ik"
            .parse::<EinsumSpecification>()
            .unwrap()
            .plan(&[lhs_type, rhs_type], &EinsumOptions::new())
            .unwrap();
        assert_eq!(plan.output_type(), Ok(output_type));
        assert_eq!(plan.infer().unwrap().1, program.instructions().len());
    }

    #[test]
    fn test_einsum_batching() {
        // Each batch item is contracted independently, with a mapped input next to a replicated one.
        let matrices = reference_input(0, &[2, 2, 3]);
        let vector = reference_input(1, &[3]);
        let first = Array::einsum("ij,j->i", [&reference_input(0, &[2, 3]), &vector]).unwrap();
        let batched = batch(
            |(matrix, vector)| Einsum::einsum("ij,j->i", [&matrix, &vector]),
            (matrices.clone(), vector.clone()),
            (BatchAxis::new(0), BatchAxis::replicated()),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        assert_eq!(batched, Array::einsum("bij,j->bi", [&matrices, &vector]).unwrap());
        assert_eq!(batched.r#type().into_owned(), ArrayType::new_static(DataType::F64, [2, 2]));
        assert_eq!(&batched.elements::<f64>().unwrap()[..2], first.elements::<f64>().unwrap().as_slice());
    }

    #[test]
    fn test_einsum_differentiation() {
        // An einsum is linear in each input, so its tangent with respect to one input is the einsum of that tangent.
        let matrix = reference_input(0, &[2, 3]);
        let vector = reference_input(1, &[3]);
        let tangent = reference_input(2, &[2, 3]);
        assert_eq!(
            differentiate_at(matrix.clone())
                .with_captures(vector.clone())
                .jvp(tangent.clone(), |matrix, vector| { Einsum::einsum("ij,j->i", [&matrix, &vector]) }),
            Ok((
                Array::einsum("ij,j->i", [&matrix, &vector]).unwrap(),
                Array::einsum("ij,j->i", [&tangent, &vector]).unwrap(),
            )),
        );

        // Reverse mode transposes every step, including the diagonal of a repeated label and a multiple step path.
        check_gradient!(
            |input| Einsum::einsum(
                "iij,jk,k->",
                [&input, &input.einsum_with("iij->ij", [])?, &input.einsum_with("iij->j", [])?]
            ),
            at = reference_input(0, &[3, 3, 3]),
            step = 1e-4,
            tolerance = 1e-5,
        );
    }

    #[test]
    fn test_einsum_array_ir() {
        // Array IR values contract through their array views, so dynamic extents that inputs share are supported.
        let extent = Dimension::Dynamic(DimensionVariable::new("n", DimensionBounds::new(1, Some(8)).unwrap()));
        let lhs_type = ArrayType::new(DataType::F64, Shape::new(vec![extent.clone(), 3.into()]));
        let rhs_type = ArrayType::new(DataType::F64, Shape::new(vec![3.into(), extent]));
        let (_, program) = DomainTracingContext::<ArrayIrEagerContext>::trace(
            |inputs: Vec<_>| Einsum::einsum("ij,jk,kj->i", inputs.iter()),
            vec![ArrayIrType::from(lhs_type.clone()), ArrayIrType::from(rhs_type.clone()), ArrayIrType::from(lhs_type)],
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[n, 3], %1:f64[3, n], %2:f64[n, 3] .
                let %3:f64[3] = dot [
                    dimensions=(lhs_contracting=[1], rhs_contracting=[0], lhs_batching=[0], rhs_batching=[1]),
                ] %1 %2
                    %4:f64[n] = dot [
                        dimensions=(lhs_contracting=[1], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                    ] %0 %3
                in (%4)
            "}
            .trim_end(),
        );
        let inputs = [reference_input(0, &[2, 3]), reference_input(1, &[3, 2]), reference_input(2, &[2, 3])];
        assert_eq!(
            program.interpret(inputs.iter().cloned().map(ArrayIrValue::Array).collect::<Vec<_>>()),
            Ok(ArrayIrValue::Array(Array::einsum("ij,jk,kj->i", inputs.iter()).unwrap())),
        );

        // An extent of one broadcasts against a dynamic extent of the same label.
        let extent = Dimension::Dynamic(DimensionVariable::new("n", DimensionBounds::new(1, Some(8)).unwrap()));
        let (_, program) = DomainTracingContext::<ArrayIrEagerContext>::trace(
            |inputs: Vec<_>| Einsum::einsum("ij,j->ij", inputs.iter()),
            vec![
                ArrayIrType::from(ArrayType::new_static(DataType::F64, [2, 1])),
                ArrayIrType::from(ArrayType::new(DataType::F64, Shape::new(vec![extent]))),
            ],
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[2, 1], %1:f64[n] .
                let %2:f64[2] = reshape [shape=[2]] %0
                    %3:f64[2, n] = dot [
                        dimensions=(lhs_contracting=[], rhs_contracting=[], lhs_batching=[], rhs_batching=[]),
                    ] %2 %1
                in (%3)
            "}
            .trim_end(),
        );
        let broadcast = [reference_input(0, &[2, 1]), reference_input(1, &[3])];
        assert_eq!(
            program.interpret(broadcast.iter().cloned().map(ArrayIrValue::Array).collect::<Vec<_>>()),
            Ok(ArrayIrValue::Array(Array::einsum("i,j->ij", [&reference_input(0, &[2]), &broadcast[1]]).unwrap())),
        );

        // A repeated label takes the diagonal of the array IR value itself, so the other extents may be dynamic, and
        // so may the extent of the repeated label when its axes share it.
        let extent = Dimension::Dynamic(DimensionVariable::new("n", DimensionBounds::new(1, Some(8)).unwrap()));
        let batched_type = ArrayType::new(DataType::F64, Shape::new(vec![extent.clone(), 2.into(), 2.into()]));
        let (_, program) = DomainTracingContext::<ArrayIrEagerContext>::trace(
            |inputs: Vec<_>| Einsum::einsum("bii->bi", inputs.iter()),
            vec![ArrayIrType::from(batched_type.clone())],
        )
        .unwrap();
        assert_eq!(program.to_string(), "");
        let batched = reference_input(0, &[3, 2, 2]);
        assert_eq!(
            program.interpret(vec![ArrayIrValue::Array(batched.clone())]),
            Ok(ArrayIrValue::Array(Array::einsum("bii->bi", [&batched]).unwrap())),
        );
        let plan = "bii->bi".parse::<EinsumSpecification>().unwrap().plan(&[batched_type], &EinsumOptions::new());
        assert_eq!(
            plan.unwrap().output_type(),
            Ok(ArrayType::new(DataType::F64, Shape::new(vec![extent.clone(), 2.into()]))),
        );
        let square_type = ArrayType::new(DataType::F64, Shape::new(vec![extent.clone(), extent, 3.into()]));
        let (_, program) = DomainTracingContext::<ArrayIrEagerContext>::trace(
            |inputs: Vec<_>| Einsum::einsum("iij->j", inputs.iter()),
            vec![ArrayIrType::from(square_type)],
        )
        .unwrap();
        assert_eq!(program.to_string(), "");
        let square = reference_input(0, &[2, 2, 3]);
        assert_eq!(
            program.interpret(vec![ArrayIrValue::Array(square.clone())]),
            Ok(ArrayIrValue::Array(Array::einsum("iij->j", [&square]).unwrap())),
        );

        // Concrete array IR values contract their array members, while first-class dimension members are rejected.
        let values = inputs.iter().cloned().map(ArrayIrValue::Array).collect::<Vec<_>>();
        assert_eq!(
            ArrayIrValue::einsum("ij,jk->ik", [&values[0], &values[1]]),
            Ok(ArrayIrValue::Array(Array::einsum("ij,jk->ik", [&inputs[0], &inputs[1]]).unwrap())),
        );
        let dimension = ArrayIrValue::<Array>::Dimension(DimensionValue::constant(2).unwrap());
        assert_eq!(
            ArrayIrValue::einsum("ij,jk->ik", [&values[0], &dimension]),
            Err(ProgramError::Type(TypeError::invalid("expected array type but got dimension type"))),
        );
    }

    #[test]
    fn test_einsum_composite() {
        let function = einsum::<ArrayContext>("ij,jk,kl->il").unwrap();
        assert_eq!(function.specification().to_string(), "ij,jk,kl->il");
        assert_eq!(function.options(), &EinsumOptions::new());

        let options = EinsumOptions::new().with_path(EinsumPathStrategy::Greedy).with_accumulation_type(DataType::F32);
        let function = einsum_with_options::<ArrayContext>("ij,jk".parse().unwrap(), options.clone()).unwrap();
        assert_eq!(function.specification().to_string(), "ij,jk");
        assert_eq!(function.options(), &options);

        // Invalid subscripts are rejected when the function is created.
        assert!(matches!(einsum::<ArrayContext>("ij,jk->ik->i"), Err(ProgramError::Type(TypeError::Invalid { .. })),));

        // The key of a named composite cannot describe a sharding, so a requested output sharding is rejected.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let sharding = Sharding::new(mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()]);
        let options = EinsumOptions::new().with_output_sharding(sharding.unwrap());
        assert!(matches!(
            einsum_with_options::<ArrayContext>("ij,jk".parse().unwrap(), options),
            Err(ProgramError::UnsupportedOperation { message })
                if message == "`einsum` composites do not support a requested output sharding; use the `Einsum` \
                    capability instead",
        ));
    }

    #[test]
    fn test_einsum_composite_call() {
        // Calls agree with the capability, whether they go through the composite or contract inline.
        let cases: [(&str, &[&[usize]]); 8] = [
            ("ij,jk,kl->il", &[&[2, 3], &[3, 4], &[4, 2]]),
            ("bij,bjk->bki", &[&[2, 2, 3], &[2, 3, 4]]),
            ("ij,jk->ik", &[&[2, 3], &[3, 4]]),
            ("ii->i", &[&[3, 3]]),
            ("iij,jk,k->", &[&[3, 3, 2], &[2, 4], &[4]]),
            ("...ij,...j->...", &[&[2, 3, 4], &[4]]),
            ("i,j,k->kji", &[&[2], &[3], &[4]]),
            ("ij->ij", &[&[2, 3]]),
        ];
        for (subscripts, shapes) in cases {
            let function = einsum::<ArrayContext>(subscripts).unwrap();
            let inputs =
                shapes.iter().enumerate().map(|(index, shape)| reference_input(index, shape)).collect::<Vec<_>>();
            assert_eq!(function.call(inputs.clone()), Array::einsum(subscripts, inputs.iter()), "{subscripts}");
        }

        // One function serves inputs of different types, planning each type signature once.
        let function = einsum::<ArrayContext>("ij,jk,kl->il").unwrap();
        for shapes in [[[2, 3], [3, 4], [4, 2]], [[4, 2], [2, 5], [5, 3]], [[2, 3], [3, 4], [4, 2]]] {
            let inputs =
                shapes.iter().enumerate().map(|(index, shape)| reference_input(index, shape)).collect::<Vec<_>>();
            assert_eq!(function.call(inputs.clone()), Array::einsum("ij,jk,kl->il", inputs.iter()));
        }
        assert_eq!(function.plans.planned.lock().unwrap().len(), 2);

        // Options apply to the contractions of the body.
        let narrow = Array::matrix(1, 2, vec![bf16::from_f32(3.0), bf16::from_f32(0.5)]).unwrap();
        let options = EinsumOptions::new().with_accumulation_type(DataType::F32);
        let function = einsum_with_options::<ArrayContext>("ij,kj,kl->il".parse().unwrap(), options).unwrap();
        assert_eq!(
            function.call(vec![narrow.clone(), narrow.clone(), narrow.clone()]),
            Ok(Array::matrix(1, 2, vec![27.75f32, 4.625]).unwrap()),
        );

        // Inputs that do not match the specification are rejected before the composite boundary.
        let function = einsum::<ArrayContext>("ij,jk,kl->il").unwrap();
        assert_eq!(
            function.call(vec![reference_input(0, &[2, 3])]),
            "ij,jk,kl->il"
                .parse::<EinsumSpecification>()
                .unwrap()
                .plan(&[ArrayType::new_static(DataType::F64, [2, 3])], &EinsumOptions::new())
                .map(|_| unreachable!()),
        );
    }

    #[test]
    fn test_einsum_composite_call_staging() {
        // Each call of a plan with multiple steps stages one named composite whose body is the decomposition, and
        // calls with equal input types share one body.
        let function = einsum::<ArrayContext>("ij,jk,kl->il").unwrap();
        let matrix_type = ArrayType::new_static(DataType::F64, [2, 2]);
        let (_, program) = ArrayContext::trace(
            |inputs: Vec<DomainTracer<ArrayContext>>| {
                let first = function.call(inputs.clone())?;
                function.call(vec![first, inputs[1].clone(), inputs[2].clone()])
            },
            vec![matrix_type.clone(), matrix_type.clone(), matrix_type],
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[2, 2], %1:f64[2, 2], %2:f64[2, 2] .
                let %3:f64[2, 2] = composite [
                    key=\"ryft.einsum.v1 {inputs = [[105, 106], [106, 107], [107, 108]], output = some([105, 108]), \
                    path = [0], accumulation_type = none, precision = [0, 0, 0]}\",
                ] %0 %1 %2 [
                    primal=^0={
                        lambda %0:f64[2, 2], %1:f64[2, 2], %2:f64[2, 2] .
                        let %3:f64[2, 2] = dot [
                            dimensions=(lhs_contracting=[0], rhs_contracting=[1], lhs_batching=[], rhs_batching=[]),
                        ] %1 %0
                            %4:f64[2, 2] = dot [
                                dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                            ] %3 %2
                        in (%4)
                    },
                ]
                    %4:f64[2, 2] = composite [
                        key=\"ryft.einsum.v1 {inputs = [[105, 106], [106, 107], [107, 108]], \
                        output = some([105, 108]), path = [0], accumulation_type = none, precision = [0, 0, 0]}\",
                    ] %3 %1 %2 [
                        primal=^0,
                    ]
                in (%4)
            "}
            .trim_end(),
        );
        let inputs = vec![reference_input(0, &[2, 2]), reference_input(1, &[2, 2]), reference_input(2, &[2, 2])];
        let first = Array::einsum("ij,jk,kl->il", inputs.iter()).unwrap();
        assert_eq!(program.interpret(inputs.clone()), Array::einsum("ij,jk,kl->il", [&first, &inputs[1], &inputs[2]]),);

        // A plan that stages one instruction contracts inline.
        let function = einsum::<ArrayContext>("ij,jk->ik").unwrap();
        let (_, program) = ArrayContext::trace(
            |inputs: Vec<DomainTracer<ArrayContext>>| function.call(inputs),
            vec![ArrayType::new_static(DataType::F64, [2, 3]), ArrayType::new_static(DataType::F64, [3, 4])],
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[2, 3], %1:f64[3, 4] .
                let %2:f64[2, 4] = dot [
                    dimensions=(lhs_contracting=[1], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                ] %0 %1
                in (%2)
            "}
            .trim_end(),
        );

        // The same function stages a composite for input types whose plan has more than one step, like the ones
        // below, where the contracted label broadcasts.
        let (_, program) = ArrayContext::trace(
            |inputs: Vec<DomainTracer<ArrayContext>>| function.call(inputs),
            vec![ArrayType::new_static(DataType::F64, [2, 3]), ArrayType::new_static(DataType::F64, [1, 4])],
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[2, 3], %1:f64[1, 4] .
                let %2:f64[2, 4] = composite [
                    key=\"ryft.einsum.v1 {inputs = [[105, 106], [106, 107]], output = some([105, 107]), path = [0], \
                    accumulation_type = none, precision = [0, 0, 0]}\",
                ] %0 %1 [
                    primal={
                        lambda %0:f64[2, 3], %1:f64[1, 4] .
                        let %2:f64[4] = reshape [shape=[4]] %1
                            %3:f64[2] = reduce [kind=sum, axes=[1]] %0
                            %4:f64[2, 4] = dot [
                                dimensions=(lhs_contracting=[], rhs_contracting=[], lhs_batching=[], rhs_batching=[]),
                            ] %3 %2
                        in (%4)
                    },
                ]
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_einsum_composite_call_partial_evaluation() {
        let function = einsum::<ArrayContext>("ij,jk,kl->il").unwrap();
        let matrix_type = ArrayType::new_static(DataType::F64, [2, 2]);
        let (_, program) = ArrayContext::trace(
            |inputs: Vec<DomainTracer<ArrayContext>>| function.call(inputs).map(|output| vec![output]),
            vec![matrix_type.clone(), matrix_type.clone(), matrix_type.clone()],
        )
        .unwrap();
        let inputs = vec![reference_input(0, &[2, 2]), reference_input(1, &[2, 2]), reference_input(2, &[2, 2])];
        let expected = function.call(inputs.clone()).unwrap();

        // Known inputs fold the whole call.
        let known = program
            .partially_evaluate(&inputs.iter().cloned().map(PartialValue::Known).collect::<Vec<_>>())
            .unwrap();
        assert_eq!(
            known.program().to_string(),
            indoc! {"
                lambda  .
                in ()
            "}
            .trim_end(),
        );
        assert_eq!(known.outputs(), &[PartialEvaluationOutput::Known(expected.clone())]);

        // One unknown input keeps the call residual, with its known inputs as constants.
        let unknown = program
            .partially_evaluate(&[
                PartialValue::Unknown(matrix_type),
                PartialValue::Known(inputs[1].clone()),
                PartialValue::Known(inputs[2].clone()),
            ])
            .unwrap();
        assert_eq!(unknown.outputs(), &[PartialEvaluationOutput::Unknown(0)]);
        assert_eq!(
            unknown.program().to_string(),
            indoc! {"
                lambda %0:f64[2, 2], %1:f64[2, 2], %2:f64[2, 2] .
                let %3:f64[2, 2] = composite [
                    key=\"ryft.einsum.v1 {inputs = [[105, 106], [106, 107], [107, 108]], output = some([105, 108]), \
                    path = [0], accumulation_type = none, precision = [0, 0, 0]}\",
                ] %0 %1 %2 [
                    primal={
                        lambda %0:f64[2, 2], %1:f64[2, 2], %2:f64[2, 2] .
                        let %3:f64[2, 2] = dot [
                            dimensions=(lhs_contracting=[0], rhs_contracting=[1], lhs_batching=[], rhs_batching=[]),
                        ] %1 %0
                            %4:f64[2, 2] = dot [
                                dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                            ] %3 %2
                        in (%4)
                    },
                ]
                in (%3)
            "}
            .trim_end(),
        );
        assert_eq!(unknown.interpret(&EagerContext::new(), &[inputs[0].clone()]), Ok(vec![expected]));
    }

    #[test]
    fn test_einsum_composite_call_batching() {
        // Batching batches the body structurally, with a mapped input next to replicated ones.
        let function = einsum::<ArrayContext>("ij,jk,k->i").unwrap();
        let lhs = reference_input(0, &[2, 2, 3]);
        let rhs = reference_input(1, &[3, 4]);
        let vector = reference_input(2, &[4]);
        assert_eq!(
            batch(
                |(lhs, rhs, vector)| function.call(vec![lhs, rhs, vector]),
                (lhs.clone(), rhs.clone(), vector.clone()),
                (BatchAxis::new(0), BatchAxis::replicated(), BatchAxis::replicated()),
                BatchAxis::new(0),
                None,
            )
            .unwrap(),
            Array::einsum("bij,jk,k->bi", [&lhs, &rhs, &vector]).unwrap(),
        );

        // Differentiation commutes with batching.
        let after_batch = differentiate_at(lhs.clone())
            .with_captures((rhs.clone(), vector.clone()))
            .gradient(|lhs, (rhs, vector)| {
                batch(
                    |(lhs, rhs, vector)| function.call(vec![lhs, rhs, vector]),
                    (lhs, rhs, vector),
                    (BatchAxis::new(0), BatchAxis::replicated(), BatchAxis::replicated()),
                    BatchAxis::new(0),
                    None,
                )?
                .einsum_with("bi->", [])
            })
            .unwrap();
        let before_batch = batch(
            |(lhs, rhs, vector)| {
                differentiate_at(lhs)
                    .with_captures((rhs, vector))
                    .gradient(|lhs, (rhs, vector)| function.call(vec![lhs, rhs, vector])?.einsum_with("i->", []))
                    .map_err(ProgramError::from)
            },
            (lhs, rhs.clone(), vector.clone()),
            (BatchAxis::new(0), BatchAxis::replicated(), BatchAxis::replicated()),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        let ones = Array::from_elements(ArrayType::new_static(DataType::F64, [2, 2]), &[1.0f64; 4]).unwrap();
        let expected = Array::einsum("bi,jk,k->bij", [&ones, &rhs, &vector]).unwrap();
        assert_eq!(after_batch, expected);
        assert_eq!(before_batch, expected);
    }

    #[test]
    fn test_einsum_composite_call_differentiation() {
        // The tangent with respect to one input is the einsum of that tangent, derived from the body.
        let function = &einsum::<ArrayContext>("ij,jk,k->i").unwrap();
        let lhs = reference_input(0, &[2, 3]);
        let rhs = reference_input(1, &[3, 4]);
        let vector = reference_input(2, &[4]);
        let tangent = reference_input(3, &[2, 3]);
        assert_eq!(
            differentiate_at(lhs.clone())
                .with_captures((rhs.clone(), vector.clone()))
                .jvp(tangent.clone(), |lhs, (rhs, vector)| function.call(vec![lhs, rhs, vector])),
            Ok((
                Array::einsum("ij,jk,k->i", [&lhs, &rhs, &vector]).unwrap(),
                Array::einsum("ij,jk,k->i", [&tangent, &rhs, &vector]).unwrap(),
            )),
        );

        // Reverse mode transposes the derived tangent program, including the diagonal of a repeated label.
        let function = &einsum::<ArrayContext>("iij,jk,k->").unwrap();
        check_gradient!(
            |input| function.call(vec![
                input.clone(),
                input.einsum_with("iij->ij", [])?,
                input.einsum_with("iij->j", [])?
            ]),
            at = reference_input(0, &[3, 3, 3]),
            step = 1e-4,
            tolerance = 1e-5,
        );

        // Second derivatives differentiate the derived rules again.
        let function = &einsum::<ArrayContext>("i,ij,j->").unwrap();
        let matrix = reference_input(1, &[3, 3]);
        let hessian_vector_product = differentiate_at(reference_input(0, &[3]))
            .with_captures(matrix.clone())
            .jvp(reference_input(2, &[3]), |vector, matrix| {
                differentiate_at(vector)
                    .with_captures(matrix)
                    .gradient(|vector, matrix| function.call(vec![vector.clone(), matrix, vector]))
                    .map_err(ProgramError::from)
            })
            .unwrap()
            .1;
        let tangent = reference_input(2, &[3]);
        let expected = Array::einsum("ij,j->i", [&matrix, &tangent])
            .unwrap()
            .add(&Array::einsum("ji,j->i", [&matrix, &tangent]).unwrap())
            .unwrap();
        assert_eq!(hessian_vector_product, expected);
    }

    #[test]
    fn test_einsum_plans_planned() {
        let plans = EinsumPlans {
            specification: "ij,jk,kl->il".parse().unwrap(),
            options: EinsumOptions::new(),
            planned: Mutex::new(VecDeque::new()),
        };
        let input_types = |extent: usize| {
            vec![
                ArrayType::new_static(DataType::F64, [extent + 1, 3]),
                ArrayType::new_static(DataType::F64, [3, 4]),
                ArrayType::new_static(DataType::F64, [4, 2]),
            ]
        };

        // A type signature is planned once, and its plan carries the output type and the instruction count.
        let first = plans.planned(&input_types(1)).unwrap();
        assert!(Arc::ptr_eq(&first, &plans.planned(&input_types(1)).unwrap()));
        assert_eq!(first.plan.path(), &[(1, 2), (0, 1)]);
        assert_eq!(first.output_type, ArrayType::new_static(DataType::F64, [2, 2]));
        assert_eq!(first.instruction_count, 2);

        // The oldest signature is evicted once the capacity is exceeded.
        for extent in 2..=EinsumPlans::CAPACITY {
            plans.planned(&input_types(extent)).unwrap();
        }
        assert_eq!(plans.planned.lock().unwrap().len(), EinsumPlans::CAPACITY);
        assert!(Arc::ptr_eq(&first, &plans.planned(&input_types(1)).unwrap()));
        plans.planned(&input_types(EinsumPlans::CAPACITY + 1)).unwrap();
        assert_eq!(plans.planned.lock().unwrap().len(), EinsumPlans::CAPACITY);
        assert!(!Arc::ptr_eq(&first, &plans.planned(&input_types(1)).unwrap()));

        // Planning errors are returned and not kept.
        assert!(plans.planned(&input_types(1)[..2]).is_err());
        assert_eq!(plans.planned.lock().unwrap().len(), EinsumPlans::CAPACITY);
    }

    #[test]
    fn test_einsum_composite_key() {
        let key =
            |subscripts: &str, options: EinsumOptions| einsum_composite_key(&subscripts.parse().unwrap(), &options);
        assert_eq!(
            key("ij,jk->ik", EinsumOptions::new()).to_string(),
            "ryft.einsum.v1 {inputs = [[105, 106], [106, 107]], output = some([105, 107]), path = [0], \
             accumulation_type = none, precision = [0, 0, 0]}",
        );
        assert_eq!(
            key(
                "...i,i",
                EinsumOptions::new()
                    .with_path(EinsumPathStrategy::Explicit(vec![(0, 1)]))
                    .with_accumulation_type(DataType::F32)
                    .with_precision(DotAlgorithmPreset::BF16_BF16_F32),
            )
            .to_string(),
            "ryft.einsum.v1 {inputs = [[-1, 105], [105]], output = none, path = [4, 0, 1], \
             accumulation_type = some(f32), precision = [2, 7]}",
        );
        assert_eq!(
            key(
                "ij,jk",
                EinsumOptions::new().with_precision(
                    DotAlgorithm::new(DataType::BF16, DataType::BF16, DataType::F32)
                        .with_component_counts(1, 2)
                        .with_primitive_operation_count(3)
                        .with_imprecise_accumulation(true),
                ),
            )
            .to_string(),
            "ryft.einsum.v1 {inputs = [[105, 106], [106, 107]], output = none, path = [0], accumulation_type = none, \
             precision = [1, bf16, bf16, f32, 1, 2, 3, true]}",
        );

        // Equal specifications and options have equal keys, and every difference in them changes the key.
        assert_eq!(key("ij,jk->ik", EinsumOptions::new()), key("ij , jk -> ik", EinsumOptions::new()));
        let keys = [
            key("ij,jk->ik", EinsumOptions::new()),
            key("ij,jk", EinsumOptions::new()),
            key("ij,jk->ki", EinsumOptions::new()),
            key("ij,jk->ik", EinsumOptions::new().with_path(EinsumPathStrategy::Optimal)),
            key("ij,jk->ik", EinsumOptions::new().with_path(EinsumPathStrategy::Greedy)),
            key("ij,jk->ik", EinsumOptions::new().with_path(EinsumPathStrategy::Unoptimized)),
            key("ij,jk->ik", EinsumOptions::new().with_path(EinsumPathStrategy::Explicit(vec![(0, 1)]))),
            key("ij,jk->ik", EinsumOptions::new().with_accumulation_type(DataType::F64)),
            key("ij,jk->ik", EinsumOptions::new().with_precision(DotAlgorithmPreset::F32_F32_F32)),
        ];
        for (index, first) in keys.iter().enumerate() {
            for second in &keys[index + 1..] {
                assert_ne!(first, second);
            }
        }
    }
}
