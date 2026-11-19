use ryft_core::{ArrayType, DataType, Pad, ProgramError, Sharding, StaticShape};

use crate::arrays::Array;
use crate::arrays::shards::ArrayShard;
use crate::lowering::signature::ExecutableSignature;
use crate::{Error, Session, ToPjrt};

// TODO(eaplatanios): Review this module.

/// Pads bounded-dynamic inputs to static executable storage and appends their logical extents. An executable compiled
/// for bounded dynamic shapes takes every such input with its static bound shape, followed by a scalar argument for
/// each logical extent, and this function produces those physical inputs in that order.
///
/// Every bounded input takes one of three paths:
///
///   - An input whose actual shape equals its physical bound is passed through as it is.
///   - A smaller input is padded on its devices with zeros after its last element along every axis, by applying
///     [`Pad`] to it through `session`. The padding is an ordinary eager operation, so its compiled program is cached
///     by the session like that of any other operation, no data moves through the host, and the padded array is
///     owned by the returned inputs alone. Nothing is retained on the source array, so the bound-sized copy is
///     released as soon as the execution that consumes it lets go of it.
///   - A zero-space input below its bound is passed through as well, because the zero-space carrier materialization
///     of the execution boundary synthesizes its carrier with the bound shape directly.
///
/// The padded arrays may still be pending when this function returns. The executable that takes them as inputs
/// waits for them, and a failure of the padding surfaces as a failure of that execution.
///
/// # Errors
///
/// Returns [`ProgramError::InvalidArgument`] if a bounded input has no static bound shape or no static runtime shape,
/// exceeds its bound, has shards in other processes while it is below its bound, or has an extent that does not fit in
/// an `i32`. Errors of the padding operation and of the uploads of the extent scalars are returned unchanged.
pub(crate) fn materialize_bounded_dynamic_inputs(
    session: &Session,
    signature: &ExecutableSignature,
    declared_types: &[ArrayType],
    actual_types: &[ArrayType],
    inputs: Vec<Array>,
) -> Result<Vec<Array>, Error> {
    assert_eq!(declared_types.len(), inputs.len());
    assert_eq!(actual_types.len(), inputs.len());
    let physical_types = signature.physical_input_types(declared_types);
    let mut physical_inputs = signature.project_inputs(inputs.as_slice());
    for logical_input_index in 0..inputs.len() {
        if !signature
            .input_dimensions()
            .iter()
            .any(|input_dimension| input_dimension.logical_input_index() == logical_input_index)
        {
            continue;
        }
        let physical_input_index = signature.input_mapping()[logical_input_index].unwrap();
        let physical_type = &physical_types[physical_input_index];
        let physical_shape = physical_type.static_shape().ok_or_else(|| ProgramError::InvalidArgument {
            message: format!("bounded-dynamic executable input {logical_input_index} has no static upper shape"),
        })?;
        let actual_shape =
            actual_types[logical_input_index].static_shape().ok_or_else(|| ProgramError::InvalidArgument {
                message: format!("runtime executable input {logical_input_index} must have a static shape"),
            })?;

        // A zero-space input below its bound carries no payload to pad. The zero-space carrier materialization that
        // follows this boundary synthesizes its all-false predicate with the bound shape directly instead.
        if actual_shape == physical_shape || physical_type.data_type().is_zero() {
            continue;
        }

        let input = &inputs[logical_input_index];
        if !input.shards().iter().all(ArrayShard::is_addressable) {
            // Padding below runs on the devices of this process alone. Reject an input with a shard in another process
            // explicitly instead of failing deep inside the execution of the padding.
            return Err(ProgramError::InvalidArgument {
                message: format!(
                    "bounded-dynamic executable input {logical_input_index} is below its bound but has \
                     non-addressable shards, so it cannot be padded on this process",
                ),
            }
            .into());
        }
        physical_inputs[physical_input_index] = pad_to_bound(session, input, &actual_shape, &physical_shape)?;
    }

    for input_dimension in signature.input_dimensions() {
        let input = &inputs[input_dimension.logical_input_index()];
        let extent = actual_types[input_dimension.logical_input_index()].shape().dimensions()[input_dimension.axis()]
            .value()
            .unwrap();
        let extent = i32::try_from(extent).map_err(|_| ProgramError::InvalidArgument {
            message: format!("runtime dimension extent {extent} does not fit in a StableHLO `i32` scalar"),
        })?;

        // The scalar is uploaded for this call alone and replicated over the mesh of its input.
        let extent_type =
            ArrayType::scalar(DataType::I32).with_sharding(Sharding::replicated(input.sharding().mesh().clone(), 0))?;
        let extent_scalar =
            Array::from_host_buffer(session, extent_type, input.mesh().clone(), None, extent.to_ne_bytes())?;
        physical_inputs.push(extent_scalar);
    }
    assert_eq!(physical_inputs.len(), signature.array_input_count());
    Ok(physical_inputs)
}

/// Pads `input`, whose shape is `actual_shape`, with zeros up to `bound_shape` on its devices, keeping its elements at
/// the origin (i.e., all padding follows the last element along every axis). The padding value is a scalar zero of the
/// data type of `input` that is uploaded to the mesh of `input`, and the padding itself is the eager [`Pad`] operation
/// of `session`, whose compiled program the session caches per input type and padding amounts.
///
/// # Errors
///
/// Returns [`ProgramError::InvalidArgument`] if the shapes have different ranks or `actual_shape` exceeds
/// `bound_shape` along some axis, and the errors of the upload and of the padding operation unchanged.
fn pad_to_bound(
    session: &Session,
    input: &Array,
    actual_shape: &StaticShape,
    bound_shape: &StaticShape,
) -> Result<Array, Error> {
    let actual_extents = actual_shape.as_slice();
    let bound_extents = bound_shape.as_slice();
    let padding = (actual_extents.len() == bound_extents.len())
        .then(|| {
            actual_extents
                .iter()
                .zip(bound_extents)
                .map(|(actual, bound)| bound.checked_sub(*actual).and_then(|padding| i64::try_from(padding).ok()))
                .collect::<Option<Vec<_>>>()
        })
        .flatten()
        .ok_or_else(|| ProgramError::InvalidArgument {
            message: format!(
                "cannot pad runtime shape `{actual_extents:?}` into bounded physical shape `{bound_extents:?}`",
            ),
        })?;
    let rank = padding.len();
    let data_type = input.data_type();
    let zero_type =
        ArrayType::scalar(data_type).with_sharding(Sharding::replicated(input.sharding().mesh().clone(), 0))?;
    let zero_bytes = vec![0u8; data_type.to_pjrt().element_size_in_bytes()?];
    let zero = Array::from_host_buffer(session, zero_type, input.mesh().clone(), None, zero_bytes)?;

    // The operation runs in the session that executes the program, which is not necessarily the session of the input.
    let input = input.clone().associate(session)?;
    Ok(input.pad(&zero, vec![0i64; rank].as_slice(), padding.as_slice(), vec![0usize; rank].as_slice())?)
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use ryft_core::{
        DataType, Device, DeviceMesh, Dimension as CoreDimension, DimensionBounds, DimensionVariable, LogicalMesh,
        MeshAxis, MeshAxisType, Shape, ShardingDimension,
    };

    use crate::FromPjrt;
    use crate::sessions::tests::{domain_mesh, replicated_vector_type};
    use crate::tests::{cpu_client, execution_client, values_to_bytes};

    use super::*;

    // Returns the type of an `f32` vector with the placement `sharding` whose extent is dynamic and at most eight.
    fn bounded_vector_type(sharding: Sharding) -> ArrayType {
        ArrayType::new(
            DataType::F32,
            Shape::new(vec![DimensionVariable::new("extent", DimensionBounds::new(0, Some(9)).unwrap()).into()]),
        )
        .with_sharding(sharding)
        .unwrap()
    }

    #[test]
    fn test_materialize_bounded_dynamic_inputs() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let declared_type = bounded_vector_type(Sharding::replicated(mesh.logical_mesh().clone(), 1));
        let signature = ExecutableSignature::new(std::slice::from_ref(&declared_type), &[]);

        // An input that is already at its bound is passed through with its storage, followed by its extent scalar.
        let exact_type = replicated_vector_type(&mesh, 8);
        let exact_values = values_to_bytes::<f32>(&[0.0; 8]);
        let exact =
            Array::from_host_buffer(&session, exact_type.clone(), mesh.clone(), None, exact_values.as_slice()).unwrap();
        let at_bound = materialize_bounded_dynamic_inputs(
            &session,
            &signature,
            std::slice::from_ref(&declared_type),
            std::slice::from_ref(&exact_type),
            vec![exact.clone()],
        )
        .unwrap();
        assert_eq!(at_bound.len(), 2);
        assert_eq!(at_bound[0], exact);
        assert_eq!(at_bound[1].copy_to_host().unwrap().elements::<i32>().unwrap(), vec![8]);

        // A smaller input is padded with zeros after its elements up to the bound, and its extent scalar records its
        // actual extent.
        let smaller_type = replicated_vector_type(&mesh, 4);
        let smaller_values = values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]);
        let smaller =
            Array::from_host_buffer(&session, smaller_type.clone(), mesh, None, smaller_values.as_slice()).unwrap();
        let padded = materialize_bounded_dynamic_inputs(
            &session,
            &signature,
            std::slice::from_ref(&declared_type),
            std::slice::from_ref(&smaller_type),
            vec![smaller.clone()],
        )
        .unwrap();
        assert_eq!(padded.len(), 2);
        assert_eq!(
            padded[0].copy_to_host().unwrap().elements::<f32>().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0],
        );
        assert_eq!(padded[1].copy_to_host().unwrap().elements::<i32>().unwrap(), vec![4]);

        // The padded array and the extent scalar are owned by the result alone. The source array retains neither, so
        // the source can still donate its buffers, and dropping the result releases both.
        assert!(padded[0].has_unique_shard_buffers());
        assert!(smaller.has_unique_shard_buffers());
        let live_arrays = session.array_statistics().live_arrays;
        drop(padded);
        assert_eq!(session.array_statistics().live_arrays, live_arrays - 2);
    }

    #[test]
    fn test_materialize_bounded_dynamic_inputs_pads_every_bounded_input() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let declared_type = bounded_vector_type(Sharding::replicated(mesh.logical_mesh().clone(), 1));
        let left_type = replicated_vector_type(&mesh, 4);
        let right_type = replicated_vector_type(&mesh, 2);
        let left_values = values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]);
        let right_values = values_to_bytes::<f32>(&[5.0, 6.0]);
        let left =
            Array::from_host_buffer(&session, left_type.clone(), mesh.clone(), None, left_values.as_slice()).unwrap();
        let right = Array::from_host_buffer(&session, right_type.clone(), mesh, None, right_values.as_slice()).unwrap();
        let declared_types = [declared_type.clone(), declared_type];
        let signature = ExecutableSignature::new(&declared_types, &[]);

        // The padded inputs come first, in input order, followed by one extent scalar per bounded dimension.
        let padded = materialize_bounded_dynamic_inputs(
            &session,
            &signature,
            &declared_types,
            &[left_type, right_type],
            vec![left, right],
        )
        .unwrap();
        assert_eq!(padded.len(), 4);
        assert_eq!(
            padded[0].copy_to_host().unwrap().elements::<f32>().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0],
        );
        assert_eq!(
            padded[1].copy_to_host().unwrap().elements::<f32>().unwrap(),
            vec![5.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        );
        assert_eq!(padded[2].copy_to_host().unwrap().elements::<i32>().unwrap(), vec![4]);
        assert_eq!(padded[3].copy_to_host().unwrap().elements::<i32>().unwrap(), vec![2]);
    }

    #[test]
    fn test_materialize_bounded_dynamic_inputs_pads_sharded_inputs() {
        let client = cpu_client(4);
        // Executable signatures record placement over explicit mesh axes only, so the sharded input uses one.
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Explicit).unwrap()]).unwrap(),
            client
                .addressable_devices()
                .unwrap()
                .iter()
                .map(|device| Device::from_pjrt(device).unwrap())
                .collect(),
        )
        .unwrap();
        let session = Session::builder().client(client).build().unwrap();
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let declared_type = bounded_vector_type(sharding.clone());
        let actual_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]))
            .with_sharding(sharding)
            .unwrap();
        let source_values = values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]);
        let source =
            Array::from_host_buffer(&session, actual_type.clone(), mesh, None, source_values.as_slice()).unwrap();
        let signature = ExecutableSignature::new(std::slice::from_ref(&declared_type), &[]);

        // The padding runs on the four devices and keeps the result sharded over them.
        let padded =
            materialize_bounded_dynamic_inputs(&session, &signature, &[declared_type], &[actual_type], vec![source])
                .unwrap();
        assert_eq!(
            padded[0].copy_to_host().unwrap().elements::<f32>().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0],
        );
        assert_eq!(padded[0].shards().len(), 4);
        assert_eq!(padded[0].shards()[0].shape(), StaticShape::new(vec![2]));
    }

    #[test]
    fn test_materialize_bounded_dynamic_inputs_skips_padding_for_zero_space_inputs() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let declared_type = ArrayType::new(
            DataType::Zero,
            Shape::new(vec![DimensionVariable::new("extent", DimensionBounds::non_negative(Some(4)).unwrap()).into()]),
        )
        .with_sharding(sharding.clone())
        .unwrap();
        let signature = ExecutableSignature::new(std::slice::from_ref(&declared_type), &[]);
        let actual_type = ArrayType::new(DataType::Zero, Shape::new(vec![CoreDimension::Static(2)]))
            .with_sharding(sharding)
            .unwrap();
        let zero = Array::from_host_buffer(&session, actual_type.clone(), mesh, None, []).unwrap();

        // A zero-space input below its bound has no payload to pad, so it is passed through as it is and only its
        // extent scalar is added.
        let materialized = materialize_bounded_dynamic_inputs(
            &session,
            &signature,
            std::slice::from_ref(&declared_type),
            std::slice::from_ref(&actual_type),
            vec![zero.clone()],
        )
        .unwrap();
        assert_eq!(materialized.len(), 2);
        assert_eq!(materialized[0], zero);
        assert_eq!(materialized[1].copy_to_host().unwrap().elements::<i32>().unwrap(), vec![2]);
    }

    #[test]
    fn test_pad_to_bound() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let r#type = ArrayType::new_static(DataType::I32, [2, 2])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 2))
            .unwrap();
        let values = values_to_bytes::<i32>(&[1, 2, 3, 4]);
        let input = Array::from_host_buffer(&session, r#type, mesh, None, values.as_slice()).unwrap();
        let actual_shape = StaticShape::new(vec![2, 2]);

        // The elements stay at the origin and zeros follow them along every axis.
        let padded = pad_to_bound(&session, &input, &actual_shape, &StaticShape::new(vec![3, 4])).unwrap();
        assert_eq!(padded.shape(), StaticShape::new(vec![3, 4]));
        assert_eq!(padded.copy_to_host().unwrap().elements::<i32>().unwrap(), vec![1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0],);

        // A bound of another rank, or one that is smaller than the input along some axis, is rejected.
        assert_eq!(
            pad_to_bound(&session, &input, &actual_shape, &StaticShape::new(vec![4])).unwrap_err().to_string(),
            "cannot pad runtime shape `[2, 2]` into bounded physical shape `[4]`",
        );
        assert_eq!(
            pad_to_bound(&session, &input, &actual_shape, &StaticShape::new(vec![3, 1]))
                .unwrap_err()
                .to_string(),
            "cannot pad runtime shape `[2, 2]` into bounded physical shape `[3, 1]`",
        );
    }
}
