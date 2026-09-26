use std::collections::BTreeMap;
use std::ops::Range;

use crate::programs::ProgramError;
use crate::programs::atoms::AtomId;
use crate::programs::instructions::Instruction;
use crate::programs::operations::Operation;
use crate::programs::references::transforms::{ReferenceTransform, ReferenceTransformPath};

/// [`ReferenceTransform`]s and dynamic-input positions applied by an access to one reference input. Binding positions
/// follow the access's base inputs, grouped by reference input in increasing input-index order and then in path order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReferenceAccessDescriptor<'t, Transform: ReferenceTransform> {
    /// Ordered transforms, empty for an access to the complete root.
    transforms: &'t [Transform],

    /// Consecutive input positions supplying this path's dynamic inputs.
    bindings: Range<usize>,
}

impl<'t, Transform: ReferenceTransform> ReferenceAccessDescriptor<'t, Transform> {
    /// Creates a new [`ReferenceAccessDescriptor`] that describes the transforms and binding range of one reference
    /// input. The owning operation derives `bindings` from its base input count and the paths on preceding reference
    /// inputs.
    ///
    /// # Parameters
    ///
    ///   - `transforms`: Transforms applied in order from the reference input.
    ///   - `bindings`: Consecutive instruction input positions supplying those transforms' dynamic inputs.
    #[inline]
    pub fn new(transforms: &'t [Transform], bindings: Range<usize>) -> Self {
        Self { transforms, bindings }
    }

    /// Returns the ordered transforms applied by this access.
    #[inline]
    pub fn transforms(&self) -> &'t [Transform] {
        self.transforms
    }

    /// Returns the consecutive input positions supplying this path's dynamic inputs.
    #[inline]
    pub fn bindings(&self) -> Range<usize> {
        self.bindings.clone()
    }
}

/// Operation family whose reference accesses expose the transform paths applied to their root inputs. Every input
/// declared by a [`ReferenceEffect::Access`](crate::ReferenceEffect::Access) must have a descriptor, including
/// whole-root accesses with empty paths. Other inputs have no descriptor. Binding inputs trail the base inputs,
/// grouped by reference input position and then by transform; operation payloads store paths rather than positions.
/// Consumers read descriptors through [`validated_reference_access_descriptors`], which enforces this layout.
pub trait ReferenceAccessOperation: Operation {
    /// Transform metadata understood by the family, with dynamic bindings in its input universe.
    type Transform: ReferenceTransform<Type = Self::Type>;

    /// Returns the number of inputs before the trailing dynamic transform bindings. This count is used only for
    /// operations with declared reference accesses. Pure operations have no binding groups and may return zero.
    fn base_input_count(&self) -> usize;

    /// Returns the [`ReferenceAccessDescriptor`] for the reference access at `input_index`,
    /// or [`None`] for a non-access.
    fn reference_access_descriptor(&self, input_index: usize)
    -> Option<ReferenceAccessDescriptor<'_, Self::Transform>>;

    /// Returns a copy of this [`ReferenceAccessOperation`] with the path of the access at `input_index` replaced.
    /// Rejects a non-access position or a path unsupported by that access. The instruction's bindings must be replaced
    /// separately in the canonical layout.
    fn with_reference_access_transforms(
        &self,
        input_index: usize,
        transforms: Vec<Self::Transform>,
    ) -> Result<Self, ProgramError>;
}

/// Returns the validated access descriptor of each input of an instruction that applies `operation` to `input_count`
/// inputs, indexed by input position. Accesses map to their [`ReferenceAccessDescriptor`] while ordinary inputs,
/// including the trailing transform bindings, map to [`None`].
///
/// Validation enforces the [`ReferenceAccessOperation`] layout contract whereby every declared
/// [`ReferenceEffect::Access`](crate::ReferenceEffect::Access) is a base input with a descriptor, no other input has
/// one, the binding groups follow the base inputs in increasing access order and exactly cover the remaining inputs,
/// and consuming accesses apply no transforms. The base-input count applies only when reference accesses exist; pure
/// operations have no binding groups and retain their ordinary input layout.
///
/// [`ProgramBuilder`](crate::ProgramBuilder) is generic over every [`Operation`] family and therefore cannot
/// check this layout when instructions are added. [`ReferenceViewAnalysis`](crate::ReferenceViewAnalysis) and
/// [`rewrite_reference_access_transforms`] validate it before using any descriptor, and every other consumer must read
/// descriptors through this function instead of calling [`ReferenceAccessOperation::reference_access_descriptor`]
/// directly, so that malformed downstream families are rejected instead of producing out-of-range bindings.
///
/// # Parameters
///
///   - `operation`: Operation whose descriptors are validated.
///   - `input_count`: Number of inputs of the instruction that applies `operation`, including transform bindings.
#[inline]
pub fn validated_reference_access_descriptors<O: ReferenceAccessOperation>(
    operation: &O,
    input_count: usize,
) -> Result<Vec<Option<ReferenceAccessDescriptor<'_, O::Transform>>>, ProgramError> {
    reference_access_layout(operation, input_count).map_err(|(_, message)| ProgramError::MalformedProgram(message))
}

/// Returns a copy of `instruction` whose reference access at `input_index` applies `transforms`, with `bindings` as
/// their dynamic inputs, in place of its current transforms and bindings. Everything else (i.e., the other inputs,
/// including the binding groups of other accesses which shift as needed, outputs, attached regions, and provenance)
/// is preserved.
///
/// Use this when a program transformation changes what an existing access selects, rather than rebuilding the
/// operation by hand (e.g., prepending an index for a newly mapped batch axis, or dropping a leading index that
/// a region boundary already applies). `bindings` are atoms of the instruction's own region, like the rest of its
/// inputs. Note that this function validates only the input layout. Input and output types are checked when the
/// returned instruction is added to a [`ProgramBuilder`](crate::ProgramBuilder).
pub fn rewrite_reference_access_transforms<O: ReferenceAccessOperation>(
    instruction: &Instruction<O>,
    input_index: usize,
    transforms: Vec<O::Transform>,
    bindings: Vec<AtomId>,
) -> Result<Instruction<O>, ProgramError> {
    let operation = instruction.operation();
    let range = validated_reference_access_descriptors(operation, instruction.inputs().len())?
        .into_iter()
        .nth(input_index)
        .flatten()
        .ok_or_else(|| {
            ProgramError::MalformedProgram(format!(
                "operation `{}` has no reference access at input {input_index}",
                operation.name(),
            ))
        })?
        .bindings();
    ReferenceTransformPath::from_transforms(&transforms, &bindings)?;
    let operation = operation.with_reference_access_transforms(input_index, transforms)?;
    let mut inputs = Vec::with_capacity(instruction.inputs().len() - range.len() + bindings.len());
    inputs.extend_from_slice(&instruction.inputs()[..range.start]);
    inputs.extend(bindings);
    inputs.extend_from_slice(&instruction.inputs()[range.end..]);
    validated_reference_access_descriptors(&operation, inputs.len())?;
    Ok(Instruction::new(operation, inputs, instruction.outputs().to_vec(), instruction.regions().to_vec())
        .with_provenance(instruction.provenance().clone()))
}

/// Validates the layout described by [`validated_reference_access_descriptors`], reporting a failure as the index of
/// the failing input together with a message that names the operation. Failures of the trailing binding count are
/// attributed to the last access, whose binding group ends the canonical layout.
pub(super) fn reference_access_layout<O: ReferenceAccessOperation>(
    operation: &O,
    input_count: usize,
) -> Result<Vec<Option<ReferenceAccessDescriptor<'_, O::Transform>>>, (usize, String)> {
    let accesses = operation.effects().accesses().collect::<BTreeMap<_, _>>();
    let malformed = |input: usize, message: String| (input, format!("operation `{}` {message}", operation.name()));
    if let Some((&input, _)) = accesses.range(input_count..).next() {
        return Err(malformed(
            input,
            format!("declares reference access at input {input} but has only {input_count} inputs"),
        ));
    }

    let base_count = operation.base_input_count();
    let mut next_binding = base_count;
    let mut last_access = None;
    let mut descriptors = Vec::with_capacity(input_count);
    for input in 0..input_count {
        let descriptor = operation.reference_access_descriptor(input);
        match (accesses.get(&input), &descriptor) {
            (Some(_), None) => {
                return Err(malformed(input, format!("does not describe reference access at input {input}")));
            }
            (None, Some(_)) => {
                return Err(malformed(input, format!("describes reference transforms at non-access input {input}")));
            }
            (None, None) => {}
            (Some(mode), Some(descriptor)) => {
                if input >= base_count {
                    return Err(malformed(
                        input,
                        format!("reference access at input {input} is outside its {base_count} base inputs"),
                    ));
                }
                let start = next_binding;
                for transform in descriptor.transforms() {
                    next_binding = next_binding.checked_add(transform.binding_count()).ok_or_else(|| {
                        malformed(input, "reference transform binding count overflows `usize`".to_string())
                    })?;
                }
                if descriptor.bindings() != (start..next_binding) {
                    return Err(malformed(
                        input,
                        format!(
                            "reference access at input {} has binding range {:?}, expected {}..{}",
                            input,
                            descriptor.bindings(),
                            start,
                            next_binding,
                        ),
                    ));
                }
                if !descriptor.transforms().is_empty() && mode.is_consuming() {
                    return Err(malformed(input, format!("consumes input {input} through a reference view")));
                }
                last_access = Some(input);
            }
        }
        descriptors.push(descriptor);
    }

    if let Some(input) = last_access
        && next_binding != input_count
    {
        return Err(malformed(
            input,
            format!("reference transforms require {next_binding} inputs but the instruction has {input_count}"),
        ));
    }

    Ok(descriptors)
}

#[cfg(test)]
pub(crate) mod tests {
    use std::borrow::Cow;
    use std::fmt::Display;

    use pretty_assertions::assert_eq;

    use crate::arrays::{ArrayIrType, ArrayReferenceTransform, ArrayType};
    use crate::kernels::AsyncCopyOperation;
    use crate::programs::effects::{EffectClasses, Effects, ReferenceAccessMode, ReferenceEffect};
    use crate::programs::provenance::{Provenance, ProvenanceScope};
    use crate::programs::references::transforms::tests::{dynamic, index};
    use crate::programs::references::transforms::{BoundReferenceTransform, ReferenceViewOverlap};
    use crate::programs::regions::{RegionId, RegionInterface};
    use crate::programs::types::TypeError;

    use super::*;

    /// Downstream-style access family that stores its descriptors directly instead of deriving them from a canonical
    /// layout, so that tests can describe layouts violating the [`ReferenceAccessOperation`] contract.
    #[derive(Clone, Debug)]
    pub(crate) struct DescribedAccess {
        /// Declared reference access, as its input index and mode, or [`None`] for a pure operation.
        pub(crate) access: Option<(usize, ReferenceAccessMode)>,

        /// Number of inputs before the trailing transform bindings.
        pub(crate) base_input_count: usize,

        /// Transforms and binding range described at each input position, with missing positions describing no access.
        pub(crate) descriptors: Vec<Option<(Vec<ArrayReferenceTransform>, Range<usize>)>>,
    }

    impl Operation for DescribedAccess {
        type Type = ArrayIrType;

        fn name(&self) -> &'static str {
            "described_access"
        }

        fn infer_output_types(
            &self,
            _inputs: &[ArrayIrType],
            _regions: &[RegionInterface<ArrayIrType>],
        ) -> Result<Vec<ArrayIrType>, TypeError> {
            Ok(Vec::new())
        }

        fn effects(&self) -> Cow<'_, Effects> {
            let access = self.access.map(|(input_index, mode)| ReferenceEffect::Access { input_index, mode });
            Cow::Owned(Effects::new(EffectClasses::NONE, access.into_iter().collect()).unwrap())
        }
    }

    impl ReferenceAccessOperation for DescribedAccess {
        type Transform = ArrayReferenceTransform;

        fn base_input_count(&self) -> usize {
            self.base_input_count
        }

        fn reference_access_descriptor(
            &self,
            input_index: usize,
        ) -> Option<ReferenceAccessDescriptor<'_, Self::Transform>> {
            let (transforms, bindings) = self.descriptors.get(input_index)?.as_ref()?;
            Some(ReferenceAccessDescriptor::new(transforms, bindings.clone()))
        }

        fn with_reference_access_transforms(
            &self,
            input_index: usize,
            transforms: Vec<Self::Transform>,
        ) -> Result<Self, ProgramError> {
            // The fixture replaces only the transforms and keeps its stored binding range.
            let mut operation = self.clone();
            let Some(Some(descriptor)) = operation.descriptors.get_mut(input_index) else {
                return Err(ProgramError::MalformedProgram(format!("no reference access at input {input_index}")));
            };
            descriptor.0 = transforms;
            Ok(operation)
        }
    }

    #[test]
    fn test_reference_access_descriptor_new() {
        let transforms = [dynamic()];
        let descriptor = ReferenceAccessDescriptor::new(&transforms, 2..3);
        assert_eq!(descriptor, ReferenceAccessDescriptor::new(&[dynamic()], 2..3));
        assert_ne!(descriptor, ReferenceAccessDescriptor::new(&[dynamic()], 1..2));
        assert_ne!(descriptor, ReferenceAccessDescriptor::new(&[index(0, 0)], 2..3));
    }

    #[test]
    fn test_reference_access_descriptor_transforms() {
        let transforms = [index(0, 1), dynamic()];
        assert_eq!(ReferenceAccessDescriptor::new(&transforms, 1..2).transforms(), &transforms);
        assert_eq!(ReferenceAccessDescriptor::<ArrayReferenceTransform>::new(&[], 1..1).transforms(), &[]);
    }

    #[test]
    fn test_reference_access_descriptor_bindings() {
        assert_eq!(ReferenceAccessDescriptor::new(&[dynamic()], 2..3).bindings(), 2..3);
        assert_eq!(ReferenceAccessDescriptor::<ArrayReferenceTransform>::new(&[], 1..1).bindings(), 1..1);
    }

    #[test]
    fn test_validated_reference_access_descriptors() {
        let operation = AsyncCopyOperation::new()
            .with_source_transforms(vec![dynamic()])
            .with_destination_transforms(vec![dynamic()]);
        assert_eq!(
            validated_reference_access_descriptors(&operation, 4),
            Ok(vec![
                Some(ReferenceAccessDescriptor::new(&[dynamic()], 2..3)),
                Some(ReferenceAccessDescriptor::new(&[dynamic()], 3..4)),
                None,
                None,
            ]),
        );

        // Downstream families follow the same contract, and pure operations keep their ordinary input layout.
        let read = DescribedAccess {
            access: Some((0, ReferenceAccessMode::Read)),
            base_input_count: 1,
            descriptors: vec![Some((vec![dynamic()], 1..2))],
        };
        assert_eq!(
            validated_reference_access_descriptors(&read, 2),
            Ok(vec![Some(ReferenceAccessDescriptor::new(&[dynamic()], 1..2)), None]),
        );
        let pure = DescribedAccess { access: None, base_input_count: 0, descriptors: Vec::new() };
        assert_eq!(validated_reference_access_descriptors(&pure, 2), Ok(vec![None, None]));
    }

    #[test]
    fn test_validated_reference_access_descriptors_rejects_malformed_layouts() {
        // A declared access needs a descriptor, and only declared accesses may have one.
        let missing = DescribedAccess {
            access: Some((1, ReferenceAccessMode::Read)),
            base_input_count: 2,
            descriptors: Vec::new(),
        };
        assert_eq!(
            validated_reference_access_descriptors(&missing, 2),
            Err(ProgramError::MalformedProgram(
                "operation `described_access` does not describe reference access at input 1".to_string(),
            )),
        );
        let extraneous = DescribedAccess {
            access: Some((0, ReferenceAccessMode::Read)),
            base_input_count: 2,
            descriptors: vec![Some((Vec::new(), 2..2)), Some((Vec::new(), 2..2))],
        };
        assert_eq!(
            validated_reference_access_descriptors(&extraneous, 2),
            Err(ProgramError::MalformedProgram(
                "operation `described_access` describes reference transforms at non-access input 1".to_string(),
            )),
        );

        // Accesses must be base inputs within the instruction.
        assert_eq!(
            validated_reference_access_descriptors(&missing, 1),
            Err(ProgramError::MalformedProgram(
                "operation `described_access` declares reference access at input 1 but has only 1 inputs".to_string(),
            )),
        );
        let outside_base = DescribedAccess {
            access: Some((1, ReferenceAccessMode::Read)),
            base_input_count: 1,
            descriptors: vec![None, Some((Vec::new(), 1..1))],
        };
        assert_eq!(
            validated_reference_access_descriptors(&outside_base, 2),
            Err(ProgramError::MalformedProgram(
                "operation `described_access` reference access at input 1 is outside its 1 base inputs".to_string(),
            )),
        );

        // Binding groups start right after the base inputs and exactly cover the remaining inputs.
        let shifted = DescribedAccess {
            access: Some((0, ReferenceAccessMode::Read)),
            base_input_count: 1,
            descriptors: vec![Some((vec![dynamic()], 2..3))],
        };
        assert_eq!(
            validated_reference_access_descriptors(&shifted, 3),
            Err(ProgramError::MalformedProgram(
                "operation `described_access` reference access at input 0 has binding range 2..3, expected 1..2"
                    .to_string(),
            )),
        );
        let whole_root = DescribedAccess {
            access: Some((0, ReferenceAccessMode::Read)),
            base_input_count: 1,
            descriptors: vec![Some((Vec::new(), 1..1))],
        };
        assert_eq!(
            validated_reference_access_descriptors(&whole_root, 2),
            Err(ProgramError::MalformedProgram(
                "operation `described_access` reference transforms require 1 inputs but the instruction has 2"
                    .to_string(),
            )),
        );

        // Consumption is a complete-root lifetime event and cannot go through a view.
        let consuming = DescribedAccess {
            access: Some((0, ReferenceAccessMode::Consume)),
            base_input_count: 1,
            descriptors: vec![Some((vec![index(0, 0)], 1..1))],
        };
        assert_eq!(
            validated_reference_access_descriptors(&consuming, 1),
            Err(ProgramError::MalformedProgram(
                "operation `described_access` consumes input 0 through a reference view".to_string(),
            )),
        );
    }

    #[test]
    fn test_validated_reference_access_descriptors_rejects_overflowing_binding_counts() {
        /// Transform declaring more bindings than any instruction can supply, so that the binding layout overflows.
        #[derive(Clone, Debug, PartialEq, Eq, Hash)]
        struct UnboundedTransform;

        impl Display for UnboundedTransform {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("unbounded")
            }
        }

        impl ReferenceTransform for UnboundedTransform {
            type Type = ArrayIrType;
            type Referent = ArrayType;

            fn binding_count(&self) -> usize {
                usize::MAX
            }

            fn validate_bindings(&self, _input: &ArrayType, _bindings: &[&ArrayIrType]) -> Result<(), TypeError> {
                Ok(())
            }

            fn output_type(&self, input: &ArrayType) -> Result<ArrayType, TypeError> {
                Ok(input.clone())
            }

            fn overlap(
                _type: &ArrayIrType,
                _lhs: &[BoundReferenceTransform<Self>],
                _rhs: &[BoundReferenceTransform<Self>],
            ) -> ReferenceViewOverlap {
                ReferenceViewOverlap::MayOverlap
            }
        }

        /// Read of input 0 through one [`UnboundedTransform`], whose bindings start after that single base input.
        #[derive(Clone, Debug)]
        struct UnboundedAccess([UnboundedTransform; 1]);

        impl Operation for UnboundedAccess {
            type Type = ArrayIrType;

            fn name(&self) -> &'static str {
                "unbounded_access"
            }

            fn infer_output_types(
                &self,
                _inputs: &[ArrayIrType],
                _regions: &[RegionInterface<ArrayIrType>],
            ) -> Result<Vec<ArrayIrType>, TypeError> {
                Ok(Vec::new())
            }

            fn effects(&self) -> Cow<'_, Effects> {
                let access = ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Read };
                Cow::Owned(Effects::new(EffectClasses::NONE, vec![access]).unwrap())
            }
        }

        impl ReferenceAccessOperation for UnboundedAccess {
            type Transform = UnboundedTransform;

            fn base_input_count(&self) -> usize {
                1
            }

            fn reference_access_descriptor(
                &self,
                input_index: usize,
            ) -> Option<ReferenceAccessDescriptor<'_, Self::Transform>> {
                (input_index == 0).then(|| ReferenceAccessDescriptor::new(&self.0, 1..1))
            }

            fn with_reference_access_transforms(
                &self,
                _input_index: usize,
                _transforms: Vec<Self::Transform>,
            ) -> Result<Self, ProgramError> {
                Ok(self.clone())
            }
        }

        assert_eq!(
            validated_reference_access_descriptors(&UnboundedAccess([UnboundedTransform]), 1),
            Err(ProgramError::MalformedProgram(
                "operation `unbounded_access` reference transform binding count overflows `usize`".to_string(),
            )),
        );
    }

    #[test]
    fn test_rewrite_reference_access_transforms() {
        let operation = AsyncCopyOperation::new()
            .with_source_transforms(vec![dynamic()])
            .with_destination_transforms(vec![dynamic()]);
        let source = AtomId::new(0);
        let destination = AtomId::new(1);
        let source_index = AtomId::new(2);
        let destination_index = AtomId::new(3);
        let provenance = Provenance::scope(ProvenanceScope::new("copy"), Provenance::unknown());
        let instruction = Instruction::new(
            operation,
            vec![source, destination, source_index, destination_index],
            vec![AtomId::new(4)],
            vec![RegionId::new(1)],
        )
        .with_provenance(provenance.clone());

        // Removing the source's bindings shifts the destination's binding group left.
        let removed = rewrite_reference_access_transforms(&instruction, 0, Vec::new(), Vec::new()).unwrap();
        assert_eq!(removed.inputs(), &[source, destination, destination_index]);
        assert_eq!(removed.operation().reference_access_descriptor(0), Some(ReferenceAccessDescriptor::new(&[], 2..2)));
        assert_eq!(
            removed.operation().reference_access_descriptor(1),
            Some(ReferenceAccessDescriptor::new(&[dynamic()], 2..3)),
        );

        // Adding bindings to the source shifts the destination's binding group right.
        let first = AtomId::new(5);
        let second = AtomId::new(6);
        let added =
            rewrite_reference_access_transforms(&instruction, 0, vec![dynamic(), dynamic()], vec![first, second])
                .unwrap();
        assert_eq!(added.inputs(), &[source, destination, first, second, destination_index]);
        assert_eq!(
            added.operation().reference_access_descriptor(0),
            Some(ReferenceAccessDescriptor::new(&[dynamic(), dynamic()], 2..4)),
        );
        assert_eq!(
            added.operation().reference_access_descriptor(1),
            Some(ReferenceAccessDescriptor::new(&[dynamic()], 4..5)),
        );

        // Everything other than the rewritten access is preserved.
        assert_eq!(added.outputs(), &[AtomId::new(4)]);
        assert_eq!(added.regions(), &[RegionId::new(1)]);
        assert_eq!(added.provenance(), &provenance);
    }

    #[test]
    fn test_rewrite_reference_access_transforms_rejects_non_access_inputs() {
        let operation = AsyncCopyOperation::new().with_source_transforms(vec![dynamic()]);
        let instruction =
            Instruction::new(operation, vec![AtomId::new(0), AtomId::new(1), AtomId::new(2)], Vec::new(), Vec::new());
        assert_eq!(
            rewrite_reference_access_transforms(&instruction, 2, Vec::new(), Vec::new()).unwrap_err(),
            ProgramError::MalformedProgram("operation `async_copy` has no reference access at input 2".to_string()),
        );
    }

    #[test]
    fn test_rewrite_reference_access_transforms_rejects_mismatched_bindings() {
        let instruction =
            Instruction::new(AsyncCopyOperation::new(), vec![AtomId::new(0), AtomId::new(1)], Vec::new(), Vec::new());
        assert_eq!(
            rewrite_reference_access_transforms(&instruction, 0, Vec::new(), vec![AtomId::new(2)]).unwrap_err(),
            ProgramError::MalformedProgram("reference transform path has 1 extra bindings".to_string()),
        );
        assert_eq!(
            rewrite_reference_access_transforms(&instruction, 0, vec![dynamic()], Vec::new()).unwrap_err(),
            ProgramError::MalformedProgram("reference transform requires 1 bindings but only 0 remain".to_string()),
        );
    }

    #[test]
    fn test_rewrite_reference_access_transforms_rejects_malformed_layouts() {
        // The input layout is validated before any descriptor is read.
        let truncated = Instruction::new(AsyncCopyOperation::new(), vec![AtomId::new(0)], Vec::new(), Vec::new());
        assert_eq!(
            rewrite_reference_access_transforms(&truncated, 0, Vec::new(), Vec::new()).unwrap_err(),
            ProgramError::MalformedProgram(
                "operation `async_copy` declares reference access at input 1 but has only 1 inputs".to_string(),
            ),
        );

        // The rewritten layout is validated as well, so a consuming access cannot gain transforms.
        let consuming = Instruction::new(
            DescribedAccess {
                access: Some((0, ReferenceAccessMode::Consume)),
                base_input_count: 1,
                descriptors: vec![Some((Vec::new(), 1..1))],
            },
            vec![AtomId::new(0)],
            Vec::new(),
            Vec::new(),
        );
        assert_eq!(
            rewrite_reference_access_transforms(&consuming, 0, vec![index(0, 0)], Vec::new()).unwrap_err(),
            ProgramError::MalformedProgram(
                "operation `described_access` consumes input 0 through a reference view".to_string(),
            ),
        );
    }

    #[test]
    fn test_reference_access_layout() {
        let operation = AsyncCopyOperation::new()
            .with_source_transforms(vec![dynamic()])
            .with_destination_transforms(vec![dynamic()]);
        assert_eq!(
            reference_access_layout(&operation, 4),
            Ok(vec![
                Some(ReferenceAccessDescriptor::new(&[dynamic()], 2..3)),
                Some(ReferenceAccessDescriptor::new(&[dynamic()], 3..4)),
                None,
                None,
            ]),
        );

        // Each failure names the input it is attributed to, and a trailing binding count mismatch is attributed to the
        // last access, whose binding group ends the canonical layout.
        assert_eq!(
            reference_access_layout(&operation, 5),
            Err((
                1,
                "operation `async_copy` reference transforms require 4 inputs but the instruction has 5".to_string(),
            )),
        );
        let missing = DescribedAccess {
            access: Some((1, ReferenceAccessMode::Read)),
            base_input_count: 2,
            descriptors: Vec::new(),
        };
        assert_eq!(
            reference_access_layout(&missing, 1),
            Err((
                1,
                "operation `described_access` declares reference access at input 1 but has only 1 inputs".to_string(),
            )),
        );
    }
}
