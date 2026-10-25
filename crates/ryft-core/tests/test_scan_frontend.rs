use std::cell::RefCell;
use std::rc::Rc;

use indoc::formatdoc;
use pretty_assertions::assert_eq;

use ryft_core::{
    Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayType, BatchAxis, BindingRegionDriver,
    Context, DataType, Domain, EagerContext, LogicalMesh, MeshAxis, MeshAxisType, NamedAxes, NamedAxis, Operation,
    ProgramError, Provenance, ProvenanceScope, Scan, ScanOptions, ShardMap, ShardMapOperation, Sharding,
    ShardingDimension, Tracer, TracingContext, Typed, ValueProjection, ValueResolution, batch, differentiate_at, scan,
};

type IrValue = ArrayIrValue<Array>;
type IrOperation = ArrayIrOperation<Array>;
type IrContext = EagerContext<IrValue, IrOperation>;

/// Two devices whose different local values must be tracked by manual variation.
fn manual_mesh() -> LogicalMesh {
    LogicalMesh::new(vec![MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap()]).unwrap()
}

#[test]
fn test_scan_frontend_propagates_manual_variation_across_carries() {
    let mesh = manual_mesh();
    let replicated = Sharding::replicated(mesh.clone(), 1);
    let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["m"])]).unwrap();
    let invariant_type = ArrayType::new_static(DataType::F32, [1]).with_sharding(replicated.clone()).unwrap();
    let varying_type = ArrayType::new_static(DataType::F32, [1])
        .with_sharding(replicated.clone().with_varying_manual_axes(["m"]).unwrap())
        .unwrap();
    let invariant = "f32[1][sharding={mesh<['m'=2:manual]>, [{}]}]";
    let varying = "f32[1][sharding={mesh<['m'=2:manual]>, [{}], varying_manual={'m'}}]";

    // Variation reaches the second carry first, then the first carry on the next trace. Entry transitions belong
    // outside the scan even at count zero, where the original contents still acquire the fixed-point metadata.
    for iteration_count in [0, 2] {
        let (_, program) = TracingContext::<IrValue, IrOperation>::trace_with_named_axes(
            |inputs: Vec<Tracer<_>>| {
                let (carries, ()) = scan(
                    |_, (_, second, source), ()| Ok(((second, source.clone(), source), ())),
                    (inputs[0].clone(), inputs[1].clone(), inputs[2].clone()),
                    (),
                    ScanOptions::new().with_iteration_count(iteration_count),
                )?;
                Ok(vec![carries.0, carries.1, carries.2])
            },
            vec![
                ArrayIrType::Array(invariant_type.clone()),
                ArrayIrType::Array(invariant_type.clone()),
                ArrayIrType::Array(varying_type.clone()),
            ],
            vec![("m".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh: mesh.clone() })],
        )
        .unwrap();
        let scan_attributes = format!("carry_count=3, iteration_count={iteration_count}, reverse=false");
        assert_eq!(
            program.to_string(),
            formatdoc! {"
                lambda %0:{invariant}, %1:{invariant}, %2:{varying} .
                let %3:{varying} = parallel_vary [axis_name=\"m\"] %0
                    %4:{varying} = parallel_vary [axis_name=\"m\"] %1
                    %5:{varying}, %6:{varying}, %7:{varying} = scan [{scan_attributes}] %3 %4 %2 [
                        body={{
                            lambda %0:i64[], %1:{varying}, %2:{varying}, %3:{varying} .
                            in (%2, %3, %3)
                        }},
                    ]
                in (%5, %6, %7)"},
        );

        let map = ShardMap::new(
            mesh.clone(),
            vec![replicated.clone(), replicated.clone(), sharded.clone()],
            vec![sharded.clone(); 3],
            Vec::new(),
        )
        .unwrap();
        let global_types = vec![
            ArrayType::new_static(DataType::F32, [1]).into(),
            ArrayType::new_static(DataType::F32, [1]).into(),
            ArrayType::new_static(DataType::F32, [2]).into(),
        ];
        let operation = ShardMapOperation::from_program(&program, global_types, map).unwrap();
        let output_type = ArrayType::new_static(DataType::F32, [2]).with_sharding(sharded.clone()).unwrap();
        let first = if iteration_count == 0 { vec![1.0f32, 1.0] } else { vec![3.0, 4.0] };
        let second = if iteration_count == 0 { vec![2.0f32, 2.0] } else { vec![3.0, 4.0] };
        assert_eq!(
            IrContext::new().bind(
                operation,
                vec![program],
                &[
                    IrValue::Array(Array::vector(vec![1.0f32]).unwrap()),
                    IrValue::Array(Array::vector(vec![2.0f32]).unwrap()),
                    IrValue::Array(Array::vector(vec![3.0f32, 4.0]).unwrap()),
                ],
            ),
            Ok(vec![
                IrValue::Array(Array::from_elements(output_type.clone(), &first).unwrap()),
                IrValue::Array(Array::from_elements(output_type.clone(), &second).unwrap()),
                IrValue::Array(Array::from_elements(output_type, &[3.0f32, 4.0]).unwrap()),
            ]),
        );
    }
}

#[test]
fn test_scan_frontend_transposes_entry_variation_outside_the_loop() {
    let mesh = manual_mesh();
    let replicated = Sharding::replicated(mesh.clone(), 1);
    let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["m"])]).unwrap();
    let stacked_sharding =
        Sharding::new(mesh.clone(), vec![ShardingDimension::Replicated, ShardingDimension::sharded(["m"])]).unwrap();
    let invariant_type = ArrayType::new_static(DataType::F32, [1]).with_sharding(replicated.clone()).unwrap();
    let varying_stack_type = ArrayType::new_static(DataType::F32, [2, 1])
        .with_sharding(Sharding::replicated(mesh.clone(), 2).with_varying_manual_axes(["m"]).unwrap())
        .unwrap();
    let invariant = "f32[1][sharding={mesh<['m'=2:manual]>, [{}]}]";
    let varying = "f32[1][sharding={mesh<['m'=2:manual]>, [{}], varying_manual={'m'}}]";
    let stack = "f32[2, 1][sharding={mesh<['m'=2:manual]>, [{}, {}], varying_manual={'m'}}]";
    let (_, program) = TracingContext::<IrValue, IrOperation>::trace_with_named_axes(
        |inputs: Vec<Tracer<_>>| {
            let (carry, ()) = scan(
                |_, carry, item| {
                    let carry = ValueProjection::<ArrayType>::into_projected(carry)?;
                    let item = ValueProjection::<ArrayType>::into_projected(item)?;
                    Ok(((carry + item).into_value(), ()))
                },
                inputs[0].clone(),
                inputs[1].clone(),
                ScanOptions::new(),
            )?;
            Ok(vec![carry])
        },
        vec![ArrayIrType::Array(invariant_type), ArrayIrType::Array(varying_stack_type)],
        vec![("m".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh: mesh.clone() })],
    )
    .unwrap();
    assert_eq!(
        program.to_string(),
        formatdoc! {"
            lambda %0:{invariant}, %1:{stack} .
            let %2:{varying} = parallel_vary [axis_name=\"m\"] %0
                %3:{varying} = scan [carry_count=1, iteration_count=2, reverse=false] %2 %1 [
                    body={{
                        lambda %0:i64[], %1:{varying}, %2:{varying} .
                        let %3:{varying} = add %1 %2
                        in (%3)
                    }},
                ]
            in (%3)"},
    );
    let transposed = program.transpose().unwrap();
    assert_eq!(
        transposed.to_string(),
        formatdoc! {"
            lambda %0:{varying} .
            let %1:{varying}, %2:{stack} = scan [carry_count=1, iteration_count=2, reverse=true] %0 [
                body={{
                    lambda %0:i64[], %1:{varying} .
                    in (%1, %1)
                }},
            ]
                %3:{invariant} = parallel_reduce [kind=sum, axis_name=\"m\", mesh=['m'=2:manual]] %1
            in (%3, %2)"},
    );

    // Each device's input slices receive its local seed. The initially replicated carry receives their sum once.
    let operation = ShardMapOperation::from_program(
        &transposed,
        vec![ArrayType::new_static(DataType::F32, [2]).into()],
        ShardMap::new(mesh, vec![sharded], vec![replicated.clone(), stacked_sharding.clone()], Vec::new()).unwrap(),
    )
    .unwrap();
    assert_eq!(
        IrContext::new().bind(
            operation,
            vec![transposed],
            &[IrValue::Array(Array::vector(vec![1.0f32, 1.0]).unwrap())],
        ),
        Ok(vec![
            IrValue::Array(
                Array::from_elements(
                    ArrayType::new_static(DataType::F32, [1]).with_sharding(replicated).unwrap(),
                    &[2.0f32],
                )
                .unwrap(),
            ),
            IrValue::Array(
                Array::from_elements(
                    ArrayType::new_static(DataType::F32, [2, 2]).with_sharding(stacked_sharding).unwrap(),
                    &[1.0f32, 1.0, 1.0, 1.0],
                )
                .unwrap(),
            ),
        ]),
    );
}

#[test]
fn test_scan_frontend_replays_entry_alignment_in_the_explicit_context() {
    /// Executes arrays through an explicit domain distinct from `Array::Domain`, recording every parent bind.
    #[derive(Clone)]
    struct RecordingContext {
        /// Parent operations observed by this explicit context.
        binds: Rc<RefCell<Vec<&'static str>>>,
    }

    impl Domain for RecordingContext {
        type Type = ArrayType;
        type Value = Array;
        type Constant = Array;
        type Operation = ArrayOperation<Array>;
    }

    impl Context for RecordingContext {
        fn lift(&self, constant: Array) -> Result<Array, ProgramError> {
            Ok(constant)
        }

        fn bind<O: Into<Self::Operation>, D: BindingRegionDriver<Array, Self::Operation>>(
            &self,
            operation: O,
            driver: D,
            inputs: &[Array],
        ) -> Result<Vec<Array>, ProgramError> {
            let operation = operation.into();
            self.binds.borrow_mut().push(operation.name());
            if matches!(operation, ArrayOperation::ParallelVary(_)) {
                let output_types = operation.infer_output_types(
                    &inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(),
                    &driver.region_interfaces(),
                )?;
                return Ok(vec![Array::from_logical_bytes(output_types[0].clone(), &inputs[0].logical_bytes())?]);
            }
            EagerContext::<Array, ArrayOperation<Array>>::new().bind(operation, driver, inputs)
        }

        fn is_eager(&self) -> bool {
            true
        }

        fn provenance(&self) -> Provenance {
            Provenance::unknown()
        }

        fn resolve(&self, value: &Array) -> ValueResolution<Array> {
            ValueResolution::Constant(value.clone())
        }

        fn invoke_with_provenance_origin<R, F: FnOnce() -> R>(&self, _origin: Provenance, function: F) -> R {
            function()
        }

        fn invoke_with_provenance_scope<R, F: FnOnce() -> R>(&self, _scope: ProvenanceScope, function: F) -> R {
            function()
        }
    }

    impl NamedAxes for RecordingContext {
        fn named_axis(&self, name: &str) -> Option<NamedAxis> {
            (name == "m").then(|| NamedAxis::Mesh { axis: 0, size: 2, mesh: manual_mesh() })
        }

        fn named_axes(&self) -> Vec<(String, NamedAxis)> {
            vec![("m".to_string(), self.named_axis("m").unwrap())]
        }
    }

    let mesh = manual_mesh();
    let invariant_type = ArrayType::scalar(DataType::F32).with_sharding(Sharding::replicated(mesh.clone(), 0)).unwrap();
    let varying_type = ArrayType::scalar(DataType::F32)
        .with_sharding(Sharding::replicated(mesh.clone(), 0).with_varying_manual_axes(["m"]).unwrap())
        .unwrap();
    let empty_stack_type = ArrayType::new_static(DataType::F32, [0])
        .with_sharding(Sharding::replicated(mesh, 1).with_varying_manual_axes(["m"]).unwrap())
        .unwrap();
    let context = RecordingContext { binds: Rc::new(RefCell::new(Vec::new())) };
    assert_eq!(
        context.scan(
            |_, carry, item| Ok((carry + item, ())),
            Array::from_elements(invariant_type, &[7.0f32]).unwrap(),
            Array::from_elements::<f32>(empty_stack_type, &[]).unwrap(),
            ScanOptions::new(),
        ),
        Ok((Array::from_elements(varying_type, &[7.0f32]).unwrap(), ())),
    );
    assert_eq!(*context.binds.borrow(), vec!["parallel_vary", "scan"]);
}

#[test]
fn test_scan_frontend_composes_batching_and_forward_differentiation() {
    let result = batch(
        |(carry, inputs, carry_tangent, input_tangents)| {
            differentiate_at((carry, inputs))
                .jvp((carry_tangent, input_tangents), |(carry, inputs)| {
                    scan(
                        |_, carry, item| {
                            let next = carry * item;
                            Ok((next.clone(), next))
                        },
                        carry,
                        inputs,
                        ScanOptions::new(),
                    )
                })
                .map_err(ProgramError::from)
        },
        (
            Array::vector(vec![1.0f64, 2.0]).unwrap(),
            Array::matrix(2, 3, vec![2.0f64, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap(),
            Array::vector(vec![1.0f64, 1.0]).unwrap(),
            Array::matrix(2, 3, vec![0.0f64; 6]).unwrap(),
        ),
        BatchAxis::new(0),
        BatchAxis::new(0),
        None,
    );
    assert_eq!(
        result,
        Ok((
            (
                Array::vector(vec![24.0f64, 420.0]).unwrap(),
                Array::matrix(2, 3, vec![2.0f64, 6.0, 24.0, 10.0, 60.0, 420.0]).unwrap(),
            ),
            (
                Array::vector(vec![24.0f64, 210.0]).unwrap(),
                Array::matrix(2, 3, vec![2.0f64, 6.0, 24.0, 5.0, 30.0, 210.0]).unwrap(),
            ),
        )),
    );
}

#[test]
fn test_scan_frontend_stacked_axis_shardings() {
    let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap();
    let input_type = ArrayType::new_static(DataType::F32, [3])
        .with_sharding(Sharding::replicated(mesh.clone(), 1))
        .unwrap();
    let output_type = ArrayType::new_static(DataType::F32, [3])
        .with_sharding(Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap())
        .unwrap();
    assert_eq!(
        scan(
            |_, (), slice| Ok(((), slice)),
            (),
            Array::from_elements(input_type, &[1f32, 2.0, 3.0]).unwrap(),
            ScanOptions::new().with_stacked_axis_shardings(vec![ShardingDimension::sharded(["x"])]),
        ),
        Ok(((), Array::from_elements(output_type, &[1f32, 2.0, 3.0]).unwrap())),
    );
}
