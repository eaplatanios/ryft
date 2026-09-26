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
    use pretty_assertions::assert_eq;

    use crate::arrays::{ArrayIrType, ArrayReferenceTransform};
    use crate::kernels::AsyncCopyOperation;
    use crate::programs::effects::{EffectClasses, Effects, ReferenceAccessMode, ReferenceEffect};
    use crate::programs::references::transforms::tests::{dynamic, index};
    use crate::programs::regions::RegionInterface;
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

        fn effects(&self) -> std::borrow::Cow<'_, Effects> {
            let access = self.access.map(|(input_index, mode)| ReferenceEffect::Access { input_index, mode });
            std::borrow::Cow::Owned(Effects::new(EffectClasses::NONE, access.into_iter().collect()).unwrap())
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
    fn test_reference_access_descriptor() {
        let transforms = [dynamic()];
        let descriptor = ReferenceAccessDescriptor::new(&transforms, 2..3);
        assert_eq!(descriptor.transforms(), &transforms);
        assert_eq!(descriptor.bindings(), 2..3);
        assert_eq!(descriptor.clone(), descriptor);
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
        let malformed =
            |message: &str| Err(ProgramError::MalformedProgram(format!("operation `described_access` {message}")));

        // A declared access needs a descriptor, and only declared accesses may have one.
        let missing = DescribedAccess {
            access: Some((1, ReferenceAccessMode::Read)),
            base_input_count: 2,
            descriptors: Vec::new(),
        };
        assert_eq!(
            validated_reference_access_descriptors(&missing, 2),
            malformed("does not describe reference access at input 1"),
        );
        let extraneous = DescribedAccess {
            access: Some((0, ReferenceAccessMode::Read)),
            base_input_count: 2,
            descriptors: vec![Some((Vec::new(), 2..2)), Some((Vec::new(), 2..2))],
        };
        assert_eq!(
            validated_reference_access_descriptors(&extraneous, 2),
            malformed("describes reference transforms at non-access input 1"),
        );

        // Accesses must be base inputs within the instruction.
        assert_eq!(
            validated_reference_access_descriptors(&missing, 1),
            malformed("declares reference access at input 1 but has only 1 inputs"),
        );
        let outside_base = DescribedAccess {
            access: Some((1, ReferenceAccessMode::Read)),
            base_input_count: 1,
            descriptors: vec![None, Some((Vec::new(), 1..1))],
        };
        assert_eq!(
            validated_reference_access_descriptors(&outside_base, 2),
            malformed("reference access at input 1 is outside its 1 base inputs"),
        );

        // Binding groups start right after the base inputs and exactly cover the remaining inputs.
        let shifted = DescribedAccess {
            access: Some((0, ReferenceAccessMode::Read)),
            base_input_count: 1,
            descriptors: vec![Some((vec![dynamic()], 2..3))],
        };
        assert_eq!(
            validated_reference_access_descriptors(&shifted, 3),
            malformed("reference access at input 0 has binding range 2..3, expected 1..2"),
        );
        let whole_root = DescribedAccess {
            access: Some((0, ReferenceAccessMode::Read)),
            base_input_count: 1,
            descriptors: vec![Some((Vec::new(), 1..1))],
        };
        assert_eq!(
            validated_reference_access_descriptors(&whole_root, 2),
            malformed("reference transforms require 1 inputs but the instruction has 2"),
        );

        // Consumption is a complete-root lifetime event and cannot go through a view.
        let consuming = DescribedAccess {
            access: Some((0, ReferenceAccessMode::Consume)),
            base_input_count: 1,
            descriptors: vec![Some((vec![index(0, 0)], 1..1))],
        };
        assert_eq!(
            validated_reference_access_descriptors(&consuming, 1),
            malformed("consumes input 0 through a reference view"),
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
        let instruction = Instruction::new(
            operation,
            vec![source, destination, source_index, destination_index],
            Vec::new(),
            Vec::new(),
        );
        let rewritten = rewrite_reference_access_transforms(&instruction, 0, Vec::new(), Vec::new()).unwrap();
        assert_eq!(rewritten.inputs(), &[source, destination, destination_index]);
        assert_eq!(rewritten.operation().reference_access_descriptor(0).unwrap().bindings(), 2..2);
        assert_eq!(rewritten.operation().reference_access_descriptor(1).unwrap().bindings(), 2..3);
        assert_eq!(rewritten.provenance(), instruction.provenance());
        assert_eq!(rewritten.outputs(), instruction.outputs());
        assert_eq!(rewritten.regions(), instruction.regions());
        assert_eq!(
            rewrite_reference_access_transforms(&instruction, 0, Vec::new(), vec![source_index])
                .unwrap_err()
                .to_string(),
            "encountered malformed program: reference transform path has 1 extra bindings",
        );
        let malformed = Instruction::new(instruction.operation().clone(), vec![source], Vec::new(), Vec::new());
        assert_eq!(
            rewrite_reference_access_transforms(&malformed, 0, Vec::new(), Vec::new()).unwrap_err().to_string(),
            "encountered malformed program: operation `async_copy` declares reference access at input 1 \
             but has only 1 inputs",
        );

        // Rewrites validate downstream layouts before reading any descriptor.
        let consuming = Instruction::new(
            DescribedAccess {
                access: Some((0, ReferenceAccessMode::Consume)),
                base_input_count: 1,
                descriptors: vec![Some((vec![index(0, 0)], 1..1))],
            },
            vec![source],
            Vec::new(),
            Vec::new(),
        );
        assert_eq!(
            rewrite_reference_access_transforms(&consuming, 0, Vec::new(), Vec::new()).unwrap_err().to_string(),
            "encountered malformed program: operation `described_access` consumes input 0 through a reference view",
        );
    }
}
