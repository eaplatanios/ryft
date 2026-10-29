use std::borrow::Cow;
use std::convert::Infallible;
use std::fmt::{Display, Formatter};
use std::sync::Arc;

use indoc::indoc;
use pretty_assertions::assert_eq;

use ryft_core::programs::transforms::{Transform, TransformArtifact};
use ryft_core::{
    Accuracy, Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayType, AttachedRegions,
    CastOperation, CastType, CompiledCallOperation, Context, DataType, DifferentiationContext, DifferentiationDriver,
    DifferentiationDual, DifferentiationError, DifferentiationPolicy, Dimension, DimensionBounds,
    DimensionFromScalarOperation, DimensionType, DimensionValue, DimensionVariable, Domain, EagerContext,
    InterpretableOperation, InterpretationDriver, Linearization, LogicalMesh, ManualVariationAlignment, NoIdentity,
    Operation, ParallelVary, Parameter, PartitionedProgram, Placeholder, Program, ProgramBuilder, ProgramError, Region,
    RegionDriver, RegionInterface, RegionRef, RegionSlot, ScanInterpretationDomain, ScanOperation, ShardingDimension,
    Sin, SinOperation, TangentActivity, Type, TypeError, Typed, Value, ValueDirectDispatch,
};

/// External transform arguments selecting one independently retained identity artifact.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ExternalArguments {
    /// User-defined semantic variant of the transform.
    variant: usize,
}

/// External transform marker defined entirely outside `ryft-core`'s library target.
struct ExternalIdentityTransform;

impl<V: Value, O: Operation<Type = V::Type>> Transform<Region<V, O>> for ExternalIdentityTransform {
    type Arguments = ExternalArguments;
    type Artifact = TransformArtifact<V, O, usize>;

    const DEFAULT_CACHE_CAPACITY: usize = 2;
}

/// A second external marker proving that marker identity namespaces otherwise identical keys and artifacts.
struct OtherExternalIdentityTransform;

impl<V: Value, O: Operation<Type = V::Type>> Transform<Region<V, O>> for OtherExternalIdentityTransform {
    type Arguments = ExternalArguments;
    type Artifact = TransformArtifact<V, O, usize>;

    const DEFAULT_CACHE_CAPACITY: usize = 1;
}

/// Builds the source program used by the downstream extension test.
fn identity_program() -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
    let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
    let input = builder.add_input(ArrayType::scalar(DataType::F64));
    builder.build(vec![input], vec![Placeholder], vec![Placeholder]).unwrap()
}

/// Derives an external identity transform while preserving its argument as metadata.
fn derive_identity(
    region: ryft_core::RegionRef<'_, Array, ArrayOperation<Array>>,
    arguments: &ExternalArguments,
) -> Result<TransformArtifact<Array, ArrayOperation<Array>, usize>, Infallible> {
    Ok(TransformArtifact::new(vec![Arc::new(region.to_program())], arguments.variant))
}

#[test]
fn test_external_region_transform_uses_public_cache_extension_point() {
    let program = identity_program();
    let first = program
        .entry_region_ref()
        .transform::<ExternalIdentityTransform, _, Infallible>(ExternalArguments { variant: 0 }, derive_identity)
        .unwrap();
    let repeated = program
        .entry_region_ref()
        .transform::<ExternalIdentityTransform, _, Infallible>(ExternalArguments { variant: 0 }, derive_identity)
        .unwrap();
    assert!(Arc::ptr_eq(&first.programs()[0], &repeated.programs()[0]));

    let distinct_arguments = program
        .entry_region_ref()
        .transform::<ExternalIdentityTransform, _, Infallible>(ExternalArguments { variant: 1 }, derive_identity)
        .unwrap();
    assert!(!Arc::ptr_eq(&first.programs()[0], &distinct_arguments.programs()[0]));
    assert_eq!(distinct_arguments.metadata(), &1);

    let other_marker = program
        .entry_region_ref()
        .transform::<OtherExternalIdentityTransform, _, Infallible>(ExternalArguments { variant: 0 }, derive_identity)
        .unwrap();
    assert!(!Arc::ptr_eq(&first.programs()[0], &other_marker.programs()[0]));

    let cloned = program.clone();
    let from_clone = cloned
        .entry_region_ref()
        .transform::<ExternalIdentityTransform, _, Infallible>(ExternalArguments { variant: 0 }, derive_identity)
        .unwrap();
    assert!(Arc::ptr_eq(&first.programs()[0], &from_clone.programs()[0]));

    let independent = identity_program();
    let from_independent = independent
        .entry_region_ref()
        .transform::<ExternalIdentityTransform, _, Infallible>(ExternalArguments { variant: 0 }, derive_identity)
        .unwrap();
    assert!(!Arc::ptr_eq(&first.programs()[0], &from_independent.programs()[0]));
}

#[test]
fn test_external_tangent_activity_uses_custom_differentiation_driver() {
    type ArrayContext = EagerContext<Array, ArrayOperation<Array>>;

    /// Downstream driver that derives fused JVPs through the public region API.
    struct Driver;

    impl RegionDriver<Array, ArrayOperation<Array>> for Driver {
        fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'r, Array, ArrayOperation<Array>>>
        where
            Array: 'r,
            ArrayOperation<Array>: 'r,
        {
            std::iter::empty()
        }
    }

    impl DifferentiationDriver<ArrayContext> for Driver {
        fn jvp_program(
            &self,
            region: RegionRef<'_, Array, ArrayOperation<Array>>,
            input_indices: &[usize],
        ) -> Result<Arc<Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>>>, DifferentiationError> {
            region.jvp_shared(input_indices)
        }

        fn linearize_program(
            &self,
            region: RegionRef<'_, Array, ArrayOperation<Array>>,
            input_indices: &[usize],
            _instantiated_outputs: Option<&[bool]>,
        ) -> Result<Linearization<Array, ArrayOperation<Array>>, DifferentiationError> {
            region.linearize_shared(input_indices)
        }

        fn partition_jvp_program(
            &self,
            _region: RegionRef<'_, Array, ArrayOperation<Array>>,
            _input_known: &[bool],
            _required_known_outputs: &[usize],
        ) -> Result<PartitionedProgram<Array, ArrayOperation<Array>>, DifferentiationError> {
            Err(ProgramError::UnsupportedOperation {
                message: "partitioning is not supported by the activity test driver".to_string(),
            }
            .into())
        }

        fn bind_jvp_operation<P: DifferentiationPolicy<ArrayContext>>(
            &self,
            _context: &DifferentiationContext<ArrayContext, P>,
            _operation: &ArrayOperation<Array>,
            _programs: Vec<Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>>>,
            _inputs: &[DifferentiationDual<Array>],
        ) -> Result<Vec<DifferentiationDual<Array>>, DifferentiationError> {
            Err(ProgramError::UnsupportedOperation {
                message: "operation binding is not supported by the activity test driver".to_string(),
            }
            .into())
        }
    }

    let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
    let first = builder.add_input(ArrayType::scalar(DataType::F64));
    let second = builder.add_input(ArrayType::scalar(DataType::F64));
    let program = builder
        .build::<Vec<Array>, Vec<Array>>(vec![second, first], vec![Placeholder; 2], vec![Placeholder; 2])
        .unwrap();
    let region = program.entry_region_ref();
    let activity: ryft_core::differentiation::TangentActivity<ArrayContext> =
        TangentActivity::new(&Driver, region, &[true, false]).unwrap();
    assert_eq!(activity.input_indices(), &[0]);
    assert_eq!(activity.output_tangent_liveness(), &[false, true]);
    assert_eq!(activity.live_tangent_slots(activity.output_tangent_liveness()), vec![1]);
    assert_eq!(
        activity.jvp_program().to_string(),
        indoc! {"
            lambda %0:f64[], %1:f64[], %2:f64[] .
            let %3:f64[] = zero [type=f64[]]
            in (%1, %0, %3, %2)
        "}
        .trim_end(),
    );

    // Swapping the two carries propagates activity from the first carry to the second on the next iteration.
    let (input_tangent_liveness, activity) =
        TangentActivity::<ArrayContext>::from_loop_body(&Driver, region, vec![true, false], 0..2, 0..2).unwrap();
    assert_eq!(input_tangent_liveness, vec![true, true]);
    assert_eq!(activity.input_indices(), &[0, 1]);
    assert_eq!(activity.output_tangent_liveness(), &[true, true]);
    assert_eq!(activity.live_tangent_slots(activity.output_tangent_liveness()), vec![0, 1]);
    assert_eq!(
        activity.jvp_program().to_string(),
        indoc! {"
            lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[] .
            in (%1, %0, %3, %2)
        "}
        .trim_end(),
    );
}

/// Downstream value that implements capabilities directly even though its domain can bind the operations that the
/// domain-binding blanket capability implementations would bind, which only the [`ValueDirectDispatch`] marker keeps
/// disjoint from those blanket implementations.
#[derive(Clone, Debug, PartialEq)]
struct DirectArray {
    /// Host array that this value wraps.
    array: Array,
}

impl Display for DirectArray {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.array, formatter)
    }
}

impl Parameter for DirectArray {}

impl Typed for DirectArray {
    type Type = ArrayType;

    fn r#type(&self) -> Cow<'_, ArrayType> {
        self.array.r#type()
    }
}

/// Downstream operation family of the [`DirectArray`] domain. It rejects guards, which eager replay never requests.
#[derive(Clone, Debug, ryft_macros::Operation)]
#[ryft(crate = "ryft_core", type(ArrayType), constant(DirectArray))]
enum DirectOperation {
    Sin(SinOperation<ArrayType>),
}

impl Value for DirectArray {
    type Dispatch = ValueDirectDispatch;
    type Domain = EagerContext<Self, DirectOperation>;

    fn domain(&self) -> Self::Domain {
        EagerContext::new()
    }
}

// The domain's operation family converts from `SinOperation`, so the blanket `Sin` implementation would also apply to
// this value (i.e., a conflicting implementation, E0119) if the value did not use `ValueDirectDispatch`.
impl Sin for DirectArray {
    fn sin_with_accuracy(&self, accuracy: Accuracy) -> Result<Self, ProgramError> {
        Ok(Self { array: self.array.sin_with_accuracy(accuracy)? })
    }
}

// Composition blanket implementations, such as `ManualVariationAlignment`, still apply to direct values, and this one
// requires a direct `ParallelVary` implementation, because the domain-binding `ParallelVary` blanket does not apply.
impl ParallelVary for DirectArray {
    fn parallel_vary(&self, axis_name: &str) -> Result<Self, ProgramError> {
        Err(ProgramError::UnsupportedOperation {
            message: format!("`parallel_vary` requires an active manual mesh axis `{axis_name}`"),
        })
    }

    fn parallel_vary_on_mesh(&self, axis_name: &str, _mesh: &LogicalMesh) -> Result<Self, ProgramError> {
        self.parallel_vary(axis_name)
    }
}

#[test]
fn test_direct_dispatch_value_implements_domain_binding_capabilities_directly() {
    let value = DirectArray { array: Array::scalar(0.5f64).unwrap() };
    let expected = DirectArray { array: Array::scalar(0.5f64.sin()).unwrap() };

    // Capability calls use the direct implementation.
    assert_eq!(value.sin(), Ok(expected.clone()));

    // The domain binds the same operation, and its interpretation bottoms out in the direct implementation.
    assert_eq!(
        value.domain().bind(SinOperation::<ArrayType>::new(), Vec::new(), std::slice::from_ref(&value)),
        Ok(vec![expected]),
    );

    // Composition blanket implementations still apply, and values without manual variation are already aligned.
    let other = DirectArray { array: Array::scalar(2.0f64).unwrap() };
    assert_eq!(
        ManualVariationAlignment::align_manual_variation(&[value.clone(), other.clone()]),
        Ok(vec![value, other]),
    );
}

#[test]
fn test_external_cast_type_owns_inference() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    static INFERENCES: AtomicUsize = AtomicUsize::new(0);

    // This independent scalar family has no dimensions or identities and needs no array-specific bounds.
    #[derive(Clone, Debug, PartialEq)]
    struct ExternalCastType;

    impl Display for ExternalCastType {
        fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("external")
        }
    }

    impl Parameter for ExternalCastType {}

    impl Type for ExternalCastType {
        type Identity = NoIdentity;
        type Refinements = ();

        fn is_compatible_with(&self, other: &Self) -> bool {
            self == other
        }

        fn is_refined_by(&self, other: &Self) -> bool {
            self == other
        }

        fn is_scalar(&self) -> bool {
            true
        }

        fn is_complex(&self) -> bool {
            false
        }
    }

    impl CastType for ExternalCastType {
        // This family has no identities and keeps the default witness selector, which returns `None`. The unused
        // `CastOperation<Self>` placeholder reuses the conversion that cast differentiation already requires.
        type IdentityWitnessOperation = CastOperation<Self>;

        fn infer_cast_output_type(
            operation: &CastOperation<Self>,
            input_types: &[Self],
        ) -> Result<(Self, Option<Self>), TypeError> {
            INFERENCES.fetch_add(1, Ordering::Relaxed);
            // The primitive validates the family-independent declaration rules, including the input count through
            // the default refinement implementation. All types in this family are equal, so an admitted signature
            // proves the relationship, including all witness declarations.
            operation.validate_declaration(input_types)?;
            Ok((operation.output_type().clone(), Some(operation.output_type().clone())))
        }
    }

    let operation = CastOperation::checked(ExternalCastType, vec![ExternalCastType, ExternalCastType]).unwrap();
    assert_eq!(operation.infer_output_types(&[ExternalCastType, ExternalCastType], &[]), Ok(vec![ExternalCastType]));
    assert_eq!(INFERENCES.load(Ordering::Relaxed), 1);
    // The primitive owns the arity diagnostic for every family through `validate_declaration`.
    assert_eq!(
        ExternalCastType::infer_cast_output_type(&operation, &[ExternalCastType]),
        Err(TypeError::invalid("expected 2 inputs but got 1")),
    );
    // Constructors reject a missing value without invoking the independent type family's inference.
    assert_eq!(
        CastOperation::checked(ExternalCastType, Vec::new()),
        Err(TypeError::invalid("`cast` requires a declared first input type")),
    );
    assert_eq!(
        CastOperation::unchecked(ExternalCastType, Vec::new()),
        Err(TypeError::invalid("`cast` requires a declared first input type")),
    );
    assert_eq!(INFERENCES.load(Ordering::Relaxed), 2);
}

#[test]
fn test_compiled_call_operation_compiled_call_with_type_signature() {
    /// Downstream call whose results are exactly its callee's results.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct ExternalCallOperation;

    impl Operation for ExternalCallOperation {
        type Type = ArrayIrType;

        fn name(&self) -> &'static str {
            "external_call"
        }

        fn region_slots(&self) -> &'static [RegionSlot] {
            const { &[RegionSlot::computation("callee")] }
        }

        fn infer_output_types(
            &self,
            _input_types: &[ArrayIrType],
            region_interfaces: &[RegionInterface<ArrayIrType>],
        ) -> Result<Vec<ArrayIrType>, TypeError> {
            Ok(region_interfaces[0].output_types().to_vec())
        }
    }

    /// Downstream operation family that implements only the required `compiled_call` constructor.
    #[derive(Clone, Debug, ryft_macros::Operation)]
    #[ryft(crate = "ryft_core", type(ArrayIrType), constant(ArrayIrValue<Array>))]
    enum ExternalOperation {
        DimensionFromScalar(DimensionFromScalarOperation),
        Call(ExternalCallOperation),
    }

    impl CompiledCallOperation<ArrayIrValue<Array>> for ExternalOperation {
        fn compiled_call(_capture_count: usize) -> Self {
            Self::Call(ExternalCallOperation)
        }
    }

    // The callee defines its result dimension itself, so the identity is local to the callee.
    let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(8)).unwrap());
    let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ExternalOperation>::new();
    let extent = builder.add_input(ArrayType::scalar(DataType::I64).into());
    let dimension = builder
        .add_instruction(
            ExternalOperation::DimensionFromScalar(DimensionFromScalarOperation::new(DimensionType::from(rows))),
            Vec::new(),
            vec![extent],
            None,
        )
        .unwrap()[0];
    let callee = builder
        .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![dimension],
            vec![Placeholder],
            vec![Placeholder],
        )
        .unwrap();

    // The default binds `compiled_call`, so the call instruction itself defines the callee-local result identity.
    let operation =
        ExternalOperation::compiled_call_with_type_signature(0, &callee.input_types(), &callee.output_types()).unwrap();
    let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ExternalOperation>::new();
    let extent = builder.add_input(ArrayType::scalar(DataType::I64).into());
    let region = builder.import_program(callee.clone());
    let output = builder.add_instruction(operation.clone(), vec![region], vec![extent], None).unwrap()[0];
    let program = builder
        .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![output], vec![Placeholder], vec![Placeholder])
        .unwrap();
    assert_eq!(program.output_types(), callee.output_types());

    // A second call cannot define the same identity again: separate calls need not produce the same extent.
    let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ExternalOperation>::new();
    let region = builder.import_program(callee);
    let first = builder.add_input(ArrayType::scalar(DataType::I64).into());
    let second = builder.add_input(ArrayType::scalar(DataType::I64).into());
    let first = builder.add_instruction(operation.clone(), vec![region], vec![first], None).unwrap()[0];
    let second = builder.add_instruction(operation, vec![region], vec![second], None).unwrap()[0];
    assert!(matches!(
        builder.build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![first, second],
            vec![Placeholder; 2],
            vec![Placeholder; 2],
        ),
        Err(ProgramError::Type(TypeError::Invalid { message, .. }))
            if message == "operation `external_call` output defines identity rows more than once in this region",
    ));
}

#[test]
fn test_external_scan_interpretation_domain() {
    type BackendValue = ArrayIrValue<Array>;
    type BackendOperation = ArrayIrOperation<Array>;
    type HostContext = EagerContext<BackendValue, BackendOperation>;

    /// Downstream passive domain and body driver, with no `Context` implementation.
    struct Backend<'o> {
        /// Body supplied to the scan application.
        body: RegionRef<'o, BackendValue, BackendOperation>,
    }

    impl Domain for Backend<'_> {
        type Type = ArrayIrType;
        type Value = BackendValue;
        type Constant = BackendValue;
        type Operation = BackendOperation;
    }

    impl ScanInterpretationDomain for Backend<'_> {
        fn iteration_count(iteration_count: &Self::Value) -> Result<usize, ProgramError> {
            HostContext::iteration_count(iteration_count)
        }

        fn iteration_index(&self, iteration: usize) -> Result<Self::Value, ProgramError> {
            HostContext::new().iteration_index(iteration)
        }

        fn empty_stack(
            &self,
            output_type: &Self::Type,
            body_input_types: &[Self::Type],
            inputs: &[Self::Value],
        ) -> Result<Self::Value, ProgramError> {
            HostContext::new().empty_stack(output_type, body_input_types, inputs)
        }

        fn read_iteration(stack: &Self::Value, iteration: usize) -> Result<Self::Value, ProgramError> {
            HostContext::read_iteration(stack, iteration)
        }

        fn write_iteration(
            &self,
            output_index: usize,
            stack: Option<Self::Value>,
            iteration_count: usize,
            axis_sharding: &ShardingDimension,
            iteration: usize,
            value: Self::Value,
        ) -> Result<Self::Value, ProgramError> {
            HostContext::new().write_iteration(output_index, stack, iteration_count, axis_sharding, iteration, value)
        }
    }

    impl RegionDriver<BackendValue, BackendOperation> for Backend<'_> {
        fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'r, BackendValue, BackendOperation>>
        where
            BackendValue: 'r,
            BackendOperation: 'r,
        {
            std::iter::once(self.body)
        }
    }

    impl<'o> InterpretationDriver<Backend<'o>> for Backend<'o> {
        fn interpret_region(
            &self,
            _context: &Backend<'o>,
            index: usize,
            inputs: Vec<BackendValue>,
        ) -> Result<Vec<BackendValue>, ProgramError> {
            self.region(index)?.to_program().interpret(inputs)
        }

        fn bind(
            &self,
            _context: &Backend<'o>,
            operation: BackendOperation,
            regions: AttachedRegions<'_, BackendValue, BackendOperation>,
            inputs: &[BackendValue],
        ) -> Result<Vec<BackendValue>, ProgramError> {
            HostContext::new().bind(operation, regions, inputs)
        }
    }

    let mut builder = ProgramBuilder::<BackendValue, BackendOperation>::new();
    let _index = builder.add_input(ArrayType::scalar(DataType::I64).into());
    let slice = builder.add_input(ArrayType::scalar(DataType::I64).into());
    let body = builder
        .build::<Vec<BackendValue>, Vec<BackendValue>>(vec![slice], vec![Placeholder, Placeholder], vec![Placeholder])
        .unwrap();
    assert_eq!(
        body.to_string(),
        indoc! {"
            lambda %0:i64[], %1:i64[] .
            in (%1)"},
    );
    let backend = Backend { body: body.entry_region_ref() };
    let length = DimensionVariable::new("length", DimensionBounds::non_negative(Some(4)).unwrap());
    let operation = ScanOperation::<ArrayIrType>::new(0, Dimension::Dynamic(length)).with_reverse(true);

    // Every logical slice is restored to its own position even though visits run in reverse order.
    let stack = BackendValue::Array(Array::vector(vec![2i64, 3, 5]).unwrap());
    let length = BackendValue::Dimension(DimensionValue::constant(3).unwrap());
    assert_eq!(operation.interpret(&backend, &backend, &[stack.clone(), length]), Ok(vec![stack]));

    // Zero trips use the external domain's empty-stack construction instead of calling the body.
    let empty = BackendValue::Array(Array::vector(Vec::<i64>::new()).unwrap());
    let length = BackendValue::Dimension(DimensionValue::constant(0).unwrap());
    assert_eq!(operation.interpret(&backend, &backend, &[empty.clone(), length]), Ok(vec![empty]));
}
