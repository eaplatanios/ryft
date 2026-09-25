//! Bounded generated interpreter cases and shrinking without additional property-testing dependencies.

use pretty_assertions::assert_eq;

use crate::arrays::{
    Array, ArrayIrOperation, ArrayIrValue, ArrayType, DataType, Dimension, DimensionBounds, DimensionType,
    DimensionValue,
};
use crate::contexts::Context;
use crate::kernels::calls::{KernelCallOperation, KernelDefinition, KernelParameter};
use crate::kernels::grids::{Grid, GridDimension, GridExecution};
use crate::kernels::mappings::{BlockMapping, BoundaryPolicy};
use crate::kernels::memory::{MaskedLoadOperation, MaskedStoreOperation};
use crate::kernels::validation::KernelParameterAccess;
use crate::operations::DimensionMulOperation;
use crate::parameters::Placeholder;
use crate::programs::{ProgramBuilder, Typed};

/// A well-typed masked copy with one repeated lane mask and independently initialized destination storage.
#[derive(Clone, Debug, PartialEq, Eq)]
struct MaskedCopyCase {
    /// Immutable source elements.
    input: Vec<i32>,
    /// Destination elements before the masked stores.
    initial: Vec<i32>,
    /// Lane mask; its nonzero length is also the logical block width.
    mask: Vec<bool>,
}

impl MaskedCopyCase {
    /// Generates a fixed bounded enumeration including empty arrays, partial tiles, and inactive masks.
    fn generate() -> Vec<Self> {
        let mut cases = Vec::new();
        for extent in 0..=5 {
            for block in 1..=3 {
                for bits in 0..(1usize << block) {
                    for pattern in 0..2 {
                        cases.push(Self {
                            input: (0..extent).map(|index| (index as i32 - 2) * (pattern + 1)).collect(),
                            initial: (0..extent).map(|index| 7 - index as i32 * (pattern + 1)).collect(),
                            mask: (0..block).map(|lane| bits & (1 << lane) != 0).collect(),
                        });
                    }
                }
            }
        }
        cases
    }

    /// Computes expected immutable values without using kernel mappings, references, or array operations.
    fn expected(&self) -> Vec<i32> {
        self.input
            .iter()
            .zip(&self.initial)
            .enumerate()
            .map(|(index, (&input, &initial))| if self.mask[index % self.mask.len()] { input } else { initial })
            .collect()
    }

    /// Constructs the same canonical typed kernel shape for each generated case.
    fn kernel(&self) -> KernelDefinition {
        let block = self.mask.len();
        let programs = self.input.len().div_ceil(block);
        let mut mapping = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let coordinate_type =
            DimensionType::new("coordinate", DimensionBounds::non_negative(Some(programs.max(1))).unwrap());
        let coordinate = mapping.add_input(coordinate_type.clone().into());
        let width = DimensionValue::constant(block).unwrap();
        let width_input = mapping.add_constant(ArrayIrValue::Dimension(width.clone()));
        let start = mapping
            .add_instruction(
                DimensionMulOperation::new(&coordinate_type, width.r#type().as_ref()).unwrap(),
                vec![],
                vec![coordinate, width_input],
                None,
            )
            .unwrap()[0];
        let mapping = BlockMapping::new(
            mapping.build(vec![start], vec![Placeholder], vec![Placeholder]).unwrap(),
            vec![block],
            BoundaryPolicy::Masked,
        )
        .unwrap();
        let r#type = ArrayType::new_static(DataType::I32, vec![self.input.len()]);
        let operation = KernelCallOperation::new(
            Grid::new(vec![GridDimension::new(Dimension::Static(programs), GridExecution::Parallel)]).unwrap(),
            vec![
                KernelParameter::new(r#type.clone(), KernelParameterAccess::ReadOnly, mapping.clone()).unwrap(),
                KernelParameter::new(r#type, KernelParameterAccess::ReadWrite, mapping).unwrap(),
            ],
        )
        .unwrap();
        KernelDefinition::trace(operation, |(references, _coordinates)| {
            let context = references[0].context();
            let mask = context.lift(ArrayIrValue::Array(Array::vector(self.mask.clone())?))?;
            let other = context.lift(ArrayIrValue::Array(Array::vector(vec![0i32; block])?))?;
            let value = context
                .bind(MaskedLoadOperation::new(), vec![], &[references[0].clone(), mask.clone(), other])?
                .remove(0);
            context.bind(MaskedStoreOperation::new(), vec![], &[references[1].clone(), value, mask])?;
            Ok(())
        })
        .unwrap()
    }

    /// Produces strictly smaller well-typed cases. Each candidate decreases extent, block width, active mask count,
    /// or integer magnitude, so repeated successful shrinking terminates without an arbitrary iteration limit.
    fn shrink(&self) -> Vec<Self> {
        let mut cases = Vec::new();
        if !self.input.is_empty() {
            let mut smaller = self.clone();
            smaller.input.pop();
            smaller.initial.pop();
            cases.push(smaller);
        }
        if self.mask.len() > 1 {
            let mut smaller = self.clone();
            smaller.mask.pop();
            cases.push(smaller);
        }
        for (index, &active) in self.mask.iter().enumerate() {
            if active {
                let mut smaller = self.clone();
                smaller.mask[index] = false;
                cases.push(smaller);
            }
        }
        for destination in [false, true] {
            let values = if destination { &self.initial } else { &self.input };
            for (index, &value) in values.iter().enumerate() {
                if value != 0 {
                    let mut smaller = self.clone();
                    let values = if destination { &mut smaller.initial } else { &mut smaller.input };
                    values[index] = if value.unsigned_abs() > 1 { value.signum() } else { 0 };
                    cases.push(smaller);
                }
            }
        }
        cases
    }

    /// Retains only reductions that reproduce the supplied failure predicate.
    fn minimize(mut self, fails: impl Fn(&Self) -> bool) -> Self {
        assert!(fails(&self));
        while let Some(smaller) = self.shrink().into_iter().find(&fails) {
            self = smaller;
        }
        self
    }
}

#[test]
fn test_masked_copy_case_generate() {
    let cases = MaskedCopyCase::generate();
    assert_eq!(cases.len(), 168);
    assert_eq!(cases, MaskedCopyCase::generate());
}

#[test]
fn test_masked_copy_case_expected() {
    let case = MaskedCopyCase { input: vec![1, 2, 3], initial: vec![4, 5, 6], mask: vec![false, true] };
    assert_eq!(case.expected(), vec![4, 2, 6]);
}

#[test]
fn test_masked_copy_case_kernel() {
    for case in MaskedCopyCase::generate() {
        let input = Array::vector(case.input.clone()).unwrap();
        let initial = Array::vector(case.initial.clone()).unwrap();
        let actual = case.kernel().interpret(vec![input.clone(), initial.clone()], 5);
        let expected = Ok(vec![Array::vector(case.expected()).unwrap()]);
        if actual != expected {
            let minimal = case.clone().minimize(|candidate| {
                candidate.kernel().interpret(
                    vec![
                        Array::vector(candidate.input.clone()).unwrap(),
                        Array::vector(candidate.initial.clone()).unwrap(),
                    ],
                    5,
                ) != Ok(vec![Array::vector(candidate.expected()).unwrap()])
            });
            assert_eq!(actual, expected, "smallest failing case: {minimal:?}");
        }
        assert_eq!(input.elements::<i32>().unwrap(), case.input);
        assert_eq!(initial.elements::<i32>().unwrap(), case.initial);
    }
}

#[test]
fn test_masked_copy_case_shrink() {
    let case = MaskedCopyCase { input: vec![2], initial: vec![0], mask: vec![true] };
    assert_eq!(
        case.shrink(),
        vec![
            MaskedCopyCase { input: vec![], initial: vec![], mask: vec![true] },
            MaskedCopyCase { input: vec![2], initial: vec![0], mask: vec![false] },
            MaskedCopyCase { input: vec![1], initial: vec![0], mask: vec![true] },
        ]
    );
}

#[test]
fn test_masked_copy_case_minimize() {
    // This deliberately incorrect implementation ignores the mask and copies every source element.
    let ignores_mask = |case: &MaskedCopyCase| case.expected() != case.input;
    let case = MaskedCopyCase { input: vec![5, 9], initial: vec![0, 0], mask: vec![false, true] };
    let minimal = case.minimize(ignores_mask);
    assert_eq!(minimal, MaskedCopyCase { input: vec![1], initial: vec![0], mask: vec![false] });
    assert!(ignores_mask(&minimal));
    assert!(minimal.shrink().iter().all(|case| !ignores_mask(case)));
}
