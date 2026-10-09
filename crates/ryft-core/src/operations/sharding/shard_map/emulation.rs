//! Reference execution of [`ShardMapOperation`] over [`Array`] values by lockstep emulation of its devices. The
//! reference backend has a single host and no device runtime, so the emulator represents every value of a `shard_map`
//! body by one value per device, indexed by the device's coordinates along the emulated manual axes in row-major mesh
//! order, and replays the body once for all devices, one instruction at a time:
//!
//!   - **Boundaries:** Every global input is split into the shard that its input sharding assigns to each device,
//!     built at exactly the local type that [`ShardMap::local_input_type`] derives from the declared global input type.
//!     An input that is unreduced along an active manual axis (whose [`Array`] stores its logical value, i.e., the
//!     pending sum, as `Reshard for Array` does) contributes that value on the device at coordinate zero along those
//!     axes and zeros elsewhere. Every global output is assembled at its declared global type from the body outputs:
//!     tiles along the active manual axes that its output sharding names are placed in mesh order, untiled invariant
//!     axes (including reduced ones) take the value of coordinate zero, and unreduced axes are summed.
//!   - **References:** The current referent of a reference input is split like an array input, and each device
//!     receives its shard as a fresh local [`ArrayReference`] root that it owns. After the body, the final local
//!     shards are assembled like an output at the input sharding and written back into the caller's reference. Along
//!     an active manual axis along which a reference is unreduced, the devices hold partial states, which are summed
//!     (as the final-state output of the discharged map, whose output sharding is the input sharding, sums them).
//!     Along an active manual axis along which a reference is otherwise replicated, every device holds its own copy,
//!     and the reference contract of `shard_map` admits only mutations that keep the copies identical, so the copy of
//!     the device at coordinate zero is written back after checking that every copy agrees with it. Distinct
//!     reference inputs must denote distinct allocations, as in reference discharge, because each one is copied in
//!     and written back independently, which would lose the writes through all but one of two aliased inputs. A
//!     reference output forwards a reference input, so it is the caller's reference itself.
//!   - **Ordinary Operations:** Bind once per device, in device order, through
//!     [`InterpretationDriver::bind`] with exactly the local values of that device. Effects such as `print`
//!     therefore occur per device, in device order.
//!   - **Collectives:** A collective over an emulated axis (i.e., `axis_index`, `parallel_reduce`, `parallel_vary`,
//!     `parallel_all_gather`, `parallel_sum_scatter`, `parallel_all_to_all`, `parallel_ragged_all_to_all`, and
//!     `parallel_permute`) is computed across the participants of each device group directly. Every other collective
//!     binds per device like an ordinary operation.
//!   - **Region Operations:** An operation whose attached region closure contains no collective over an emulated axis
//!     binds per device, so its regions run through the ordinary eager rules. Otherwise, its regions run in lockstep:
//!     a `while` loop iterates until every device's predicate is false, a `condition` runs the branch that every
//!     device selects, a `scan` iterates over its static length (passing reference carries and stacks through, as its
//!     eager rule does), `custom_function`, `linear_call`, and `rematerialize` replay their primal region, and a nested
//!     `shard_map` extends the device set by its own active manual axes. Devices whose `while` or `condition`
//!     predicates diverge cannot run such regions in lockstep, and every other region operation that needs lockstep
//!     execution (including a `scan` with a dynamic length) is rejected with [`ProgramError::UnsupportedOperation`].
//!
//! These rejections are deliberate limitations of the emulation rather than of `shard_map`. Running a region
//! operation in lockstep requires knowing how its regions are sequenced (e.g., how many times and with which inputs),
//! which the emulator knows only for the operations above: the eager rule of any other region operation runs its
//! regions on one device's values at a time, which cannot compute a collective across devices. The trip count of a
//! dynamic-length `scan` is a per-device value that the devices could disagree on, like a divergent predicate, and
//! supporting it would add a consensus check for a case that static-length scans already cover. Backends with a device
//! runtime execute all of these bodies.

use crate::arrays::{
    Array, ArrayAddressing, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayReference, ArrayType, Dimension, Sharding,
    ShardingDimension,
};
use crate::contexts::{Domain, EagerContext};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::check_count;
use crate::operations::collectives::CollectiveMode;
use crate::operations::collectives::axis_index::AxisIndexOperation;
use crate::operations::collectives::parallel_all_gather::ParallelAllGatherOperation;
use crate::operations::collectives::parallel_all_to_all::ParallelAllToAllOperation;
use crate::operations::collectives::parallel_permute::ParallelPermuteOperation;
use crate::operations::collectives::parallel_ragged_all_to_all::ParallelRaggedAllToAllOperation;
use crate::operations::collectives::parallel_reduce::ParallelReduceOperation;
use crate::operations::collectives::parallel_sum_scatter::ParallelSumScatterOperation;
use crate::operations::collectives::parallel_vary::ParallelVaryOperation;
use crate::operations::constants::zero::Zero;
use crate::operations::control_flow::condition::{CONDITION_OPERATION_NAME, ConditionOperation};
use crate::operations::control_flow::scan::{
    SCAN_OPERATION_NAME, ScanOperation, read_scan_iteration, stacked_scan_type, write_scan_iteration,
};
use crate::operations::control_flow::r#while::{WhileOperation, WhilePredicate};
use crate::operations::custom_functions::operations::CUSTOM_FUNCTION_OPERATION_NAME;
use crate::operations::differentiation::linear_call::LINEAR_CALL_OPERATION_NAME;
use crate::operations::differentiation::rematerialize::REMATERIALIZE_OPERATION_NAME;
use crate::operations::reductions::{Reduce, ReductionKind};
use crate::programs::{
    Concretizable, EmptyRegionDriver, Instruction, Operation, OperationPayloadProjection, ProgramError, RegionRef,
    RegionReplayMappings, ReplayRegionDriver, Type, TypeError, Typed,
};

use super::{SHARD_MAP_OPERATION_NAME, ShardMap, ShardMapError, ShardMapOperation, boundary_array_type};

/// Values of one program atom on every emulated device, in row-major device order.
type DeviceValues = Vec<ArrayIrValue<Array>>;

/// Interprets `operation` over the global `inputs` by emulating each of its devices (refer to the
/// [module documentation](self) for the emulation model).
///
/// # Errors
///
/// Returns the reference contract errors of a body with reference inputs or outputs (refer to the
/// ``# References Under `shard_map` `` section of the [`shard_map`](super) module documentation),
/// [`ProgramError::UnsupportedOperation`] for the region operations that the emulator cannot run in lockstep, and the
/// errors of splitting the inputs, of binding the body's operations, of computing its collectives, and of assembling
/// the outputs and writing back the references otherwise.
pub(super) fn interpret_shard_map<C, D>(
    operation: &ShardMapOperation,
    context: &C,
    driver: &D,
    inputs: &[ArrayIrValue<Array>],
) -> Result<Vec<ArrayIrValue<Array>>, ProgramError>
where
    C: Domain<
            Type = ArrayIrType,
            Value = ArrayIrValue<Array>,
            Constant = ArrayIrValue<Array>,
            Operation: OperationPayloadProjection,
        >,
    D: InterpretationDriver<C>,
{
    operation.validate_region_count(driver.region_count())?;
    let emulator = ShardMapEmulator { context, driver, grid: ManualDeviceGrid::default() };
    let outputs = emulator.run_shard_map(
        operation,
        driver.region(0)?,
        inputs.iter().map(|input| vec![input.clone()]).collect(),
    )?;
    Ok(outputs.into_iter().map(|mut output| output.remove(0)).collect())
}

/// Devices of one lockstep emulation, identified by their coordinates along the emulated manual axes. Device indices
/// enumerate these coordinates in row-major order, so a nested grid that appends the axes of an inner manual region
/// keeps every device of the outer grid as one contiguous block of inner devices.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct ManualDeviceGrid {
    /// Names and sizes of the emulated manual axes, outermost first.
    axes: Vec<(String, usize)>,
}

impl ManualDeviceGrid {
    /// Returns this grid extended by the provided innermost `axes`.
    fn with_axes(&self, axes: &[(String, usize)]) -> Self {
        Self { axes: self.axes.iter().chain(axes).cloned().collect() }
    }

    /// Returns the number of devices of this grid, which is one for a grid without axes.
    fn device_count(&self) -> usize {
        self.axes.iter().map(|(_, size)| size).product()
    }

    /// Returns the position of the emulated axis named `axis_name`, if any.
    fn axis(&self, axis_name: &str) -> Option<usize> {
        self.axes.iter().position(|(name, _)| name == axis_name)
    }

    /// Returns the coordinates of `device` along every axis of this grid.
    fn coordinates(&self, device: usize) -> Vec<usize> {
        let mut coordinates = vec![0; self.axes.len()];
        let mut remainder = device;
        for (coordinate, (_, size)) in coordinates.iter_mut().zip(&self.axes).rev() {
            *coordinate = remainder % size;
            remainder /= size;
        }
        coordinates
    }

    /// Returns the index of the device at `coordinates`.
    fn device(&self, coordinates: &[usize]) -> usize {
        coordinates
            .iter()
            .zip(&self.axes)
            .fold(0, |device, (coordinate, (_, size))| device * size + coordinate)
    }

    /// Returns the values that `function` computes for every device of this grid, in device order.
    fn map_devices<F: FnMut(usize) -> Result<Array, ProgramError>>(
        &self,
        mut function: F,
    ) -> Result<DeviceValues, ProgramError> {
        (0..self.device_count()).map(|device| Ok(ArrayIrValue::Array(function(device)?))).collect()
    }

    /// Returns the devices that participate with `device` in a collective over the axis at position `axis`, ordered by
    /// their positions in the collective, together with the position of `device` among them. Without participant
    /// groups, the participants are the devices that differ from `device` only along `axis`, positioned by their
    /// coordinates. With participant groups, they are the members of the group that contains the coordinate of
    /// `device`, positioned by their order in that group.
    fn participants(
        &self,
        device: usize,
        axis: usize,
        axis_index_groups: Option<&[Vec<usize>]>,
    ) -> Result<(Vec<usize>, usize), ProgramError> {
        let (axis_name, axis_size) = &self.axes[axis];
        let coordinates = self.coordinates(device);
        let members = match axis_index_groups {
            None => (0..*axis_size).collect::<Vec<_>>(),
            Some(groups) => groups
                .iter()
                .find(|group| group.contains(&coordinates[axis]))
                .filter(|group| group.iter().all(|member| member < axis_size))
                .cloned()
                .ok_or_else(|| ProgramError::InvalidArgument {
                    message: format!(
                        "the participant groups `{groups:?}` of a collective over manual axis `{axis_name}` do not \
                         partition its {axis_size} coordinates",
                    ),
                })?,
        };
        let position = members.iter().position(|member| *member == coordinates[axis]).unwrap();
        let devices = members
            .iter()
            .map(|member| {
                let mut member_coordinates = coordinates.clone();
                member_coordinates[axis] = *member;
                self.device(&member_coordinates)
            })
            .collect();
        Ok((devices, position))
    }
}

/// Lockstep emulator for the devices of a [`ManualDeviceGrid`]. It replays regions over [`DeviceValues`] and binds
/// ordinary operations per device through the [`InterpretationDriver`] of the `shard_map` application being
/// interpreted.
struct ShardMapEmulator<'e, C, D> {
    /// Interpretation [`Domain`] through which ordinary operations are bound.
    context: &'e C,

    /// [`InterpretationDriver`] of the interpreted `shard_map` application, which binds ordinary operations.
    driver: &'e D,

    /// Devices of this emulation.
    grid: ManualDeviceGrid,
}

impl<C, D> ShardMapEmulator<'_, C, D>
where
    C: Domain<
            Type = ArrayIrType,
            Value = ArrayIrValue<Array>,
            Constant = ArrayIrValue<Array>,
            Operation: OperationPayloadProjection,
        >,
    D: InterpretationDriver<C>,
{
    /// Runs `operation` with the attached `body` over the global `inputs` of every device of this emulation and returns
    /// its global outputs on every device. The body runs on the grid of this emulation extended by the active manual
    /// axes of `operation`.
    fn run_shard_map(
        &self,
        operation: &ShardMapOperation,
        body: RegionRef<'_, ArrayIrValue<Array>, C::Operation>,
        inputs: Vec<DeviceValues>,
    ) -> Result<Vec<DeviceValues>, ProgramError> {
        if operation.global_input_types().iter().chain(operation.global_output_types()).any(Type::is_reference) {
            operation.validate_reference_body(body)?;
        }
        let shard_map = operation.shard_map();
        let manual_axes = shard_map
            .manual_axes()
            .iter()
            .map(|axis| {
                let size = shard_map.mesh().axis_size(axis).unwrap();
                match self.grid.axis(axis) {
                    None => Ok((axis.clone(), size)),
                    Some(_) => Err(ProgramError::MalformedProgram(format!(
                        "`{SHARD_MAP_OPERATION_NAME}` makes manual axis `{axis}` manual again inside an enclosing \
                         manual region over it",
                    ))),
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let local_grid = ManualDeviceGrid { axes: manual_axes.clone() };
        let local_device_count = local_grid.device_count();
        check_count!("input", inputs, operation.global_input_types().len(), ProgramError);

        // Every reference input is copied into fresh local roots and written back independently below, so two inputs
        // that denote the same allocation would lose the writes through one of them. They are rejected, as reference
        // discharge rejects them, per enclosing device, whose reference inputs are its own local roots.
        for device in 0..self.grid.device_count() {
            let allocations = inputs
                .iter()
                .map(|values| match &values[device] {
                    ArrayIrValue::Reference(reference) => Some(reference.id()),
                    _ => None,
                })
                .collect::<Vec<_>>();
            for (second_input_index, allocation) in allocations.iter().enumerate() {
                if allocation.is_some()
                    && let Some(first_input_index) =
                        allocations[..second_input_index].iter().position(|candidate| candidate == allocation)
                {
                    return Err(ShardMapError::RepeatedReferenceInputAllocation {
                        first_input_index,
                        second_input_index,
                    }
                    .into());
                }
            }
        }

        // Split every global input of every enclosing device, or the current referent of every reference input, into
        // the shards of its local devices. Each local device owns a fresh local root that holds its shard of a
        // reference input, which is recorded together with the caller's reference for the write-back below.
        let mut body_inputs = Vec::with_capacity(inputs.len());
        let mut reference_inputs = Vec::new();
        for (index, values) in inputs.iter().enumerate() {
            let global_type = boundary_array_type(&operation.global_input_types()[index])?;
            let local_type = shard_map.local_input_type(index, global_type)?;
            let mut local_values = Vec::with_capacity(values.len() * local_device_count);
            for value in values {
                let referent;
                let array = match value {
                    ArrayIrValue::Reference(reference) => {
                        referent = reference.read()?;
                        &referent
                    }
                    value => array_value(value, SHARD_MAP_OPERATION_NAME)?,
                };
                let shards = (0..local_device_count)
                    .map(|local_device| {
                        let coordinates = local_grid.coordinates(local_device);
                        split_input(shard_map, index, array, &local_type, &local_grid, &coordinates)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                match value {
                    ArrayIrValue::Reference(reference) => {
                        let roots = shards.into_iter().map(ArrayReference::new).collect::<Vec<_>>();
                        local_values.extend(roots.iter().cloned().map(ArrayIrValue::Reference));
                        reference_inputs.push((index, reference, roots));
                    }
                    _ => local_values.extend(shards.into_iter().map(ArrayIrValue::Array)),
                }
            }
            body_inputs.push(local_values);
        }

        let emulator =
            ShardMapEmulator { context: self.context, driver: self.driver, grid: self.grid.with_axes(&manual_axes) };
        let body_outputs = emulator.run_region(body, body_inputs)?;
        check_count!("output", body_outputs, operation.global_output_types().len(), ProgramError);

        // Write the final local shards of every reference input back into the caller's reference. Along an active
        // manual axis along which a reference is unreduced, the devices hold partial states, which assembling sums.
        // Along an active manual axis along which it is otherwise replicated, every device holds its own copy, which
        // the reference contract keeps identical (it admits only mutations with invariant values and indices), so
        // assembling takes the copy of the device at coordinate zero, after checking that every other copy agrees with
        // it.
        for (index, reference, roots) in reference_inputs {
            let shards = roots
                .iter()
                .map(|root| Ok(ArrayIrValue::Array(root.read()?)))
                .collect::<Result<Vec<_>, ProgramError>>()?;
            let reference_type = reference.r#type();
            let sharding = &shard_map.in_shardings()[index];
            validate_replicated_copies(sharding, &local_grid, &shards, index)?;
            reference.write(assemble(sharding, reference_type.referent(), &local_grid, &shards)?)?;
        }

        // Assemble every global output of every enclosing device from the body outputs of its local devices. A
        // reference output forwards a reference input, so it is the caller's reference itself.
        body_outputs
            .iter()
            .enumerate()
            .map(|(index, values)| {
                if let Some(forwarded) = operation.output_forwarding()[index] {
                    return Ok(inputs[forwarded].clone());
                }
                let global_type = <&ArrayType>::try_from(&operation.global_output_types()[index])?;
                let sharding = &shard_map.out_shardings()[index];
                values
                    .chunks(local_device_count)
                    .map(|values| Ok(ArrayIrValue::Array(assemble(sharding, global_type, &local_grid, values)?)))
                    .collect()
            })
            .collect()
    }

    /// Replays `region` over the provided per-device `inputs` in lockstep and returns its per-device outputs.
    fn run_region(
        &self,
        region: RegionRef<'_, ArrayIrValue<Array>, C::Operation>,
        inputs: Vec<DeviceValues>,
    ) -> Result<Vec<DeviceValues>, ProgramError> {
        // One replay mapping scope serves every instruction of this region replay, as in ordinary region replay.
        let mappings = RegionReplayMappings::new();
        let device_count = self.grid.device_count();
        region.interpret_with(
            inputs,
            |_, constant| Ok(vec![constant.clone(); device_count]),
            |instruction, inputs| self.run_instruction(region, &mappings, instruction, inputs),
        )
    }

    /// Runs one `instruction` of `region` on every device and returns its per-device outputs.
    fn run_instruction(
        &self,
        region: RegionRef<'_, ArrayIrValue<Array>, C::Operation>,
        mappings: &RegionReplayMappings<ArrayIrValue<Array>, C::Operation>,
        instruction: &Instruction<C::Operation>,
        inputs: &[DeviceValues],
    ) -> Result<Vec<DeviceValues>, ProgramError> {
        let operation = instruction.operation();
        if let Some(outputs) = self.run_collective(operation, inputs)? {
            return Ok(outputs);
        }
        if !self.uses_emulated_axes(region, instruction)? {
            return self.bind_per_device(region, mappings, instruction, inputs);
        }

        // The attached regions communicate across the emulated devices, so they must run in lockstep.
        let regions = instruction.regions().iter().map(|id| region.with_id(*id)).collect::<Result<Vec<_>, _>>()?;
        operation.validate_region_count(regions.len())?;
        if let Some(operation) = operation.projected_payload::<WhileOperation<ArrayIrType>>() {
            return self.run_while(operation, regions[0], regions[1], inputs);
        }
        if operation.projected_payload::<ConditionOperation<ArrayIrType>>().is_some() {
            return self.run_condition(&regions, inputs);
        }
        if let Some(operation) = operation.projected_payload::<ScanOperation<ArrayIrType>>() {
            return self.run_scan(operation, regions[0], inputs);
        }
        if let Some(operation) = operation.projected_payload::<ShardMapOperation>() {
            return self.run_shard_map(operation, regions[0], inputs.to_vec());
        }
        // The eager rules of these operations replay their primal region (i.e., region zero) over all of their inputs.
        if matches!(
            operation.name(),
            CUSTOM_FUNCTION_OPERATION_NAME | LINEAR_CALL_OPERATION_NAME | REMATERIALIZE_OPERATION_NAME,
        ) {
            return self.run_region(regions[0], inputs.to_vec());
        }
        Err(ProgramError::UnsupportedOperation {
            message: format!(
                "`{SHARD_MAP_OPERATION_NAME}` interpretation cannot run `{}` in lockstep across devices, which its \
                 attached regions require because they use collectives over the manual axes",
                operation.name(),
            ),
        })
    }

    /// Binds the operation of `instruction` once per device, in device order, with that device's inputs and the
    /// attached regions of `instruction`.
    fn bind_per_device(
        &self,
        region: RegionRef<'_, ArrayIrValue<Array>, C::Operation>,
        mappings: &RegionReplayMappings<ArrayIrValue<Array>, C::Operation>,
        instruction: &Instruction<C::Operation>,
        inputs: &[DeviceValues],
    ) -> Result<Vec<DeviceValues>, ProgramError> {
        let device_count = self.grid.device_count();
        let mut outputs = vec![Vec::with_capacity(device_count); instruction.outputs().len()];
        for device in 0..device_count {
            let device_inputs = inputs.iter().map(|values| values[device].clone()).collect::<Vec<_>>();
            let regions = ReplayRegionDriver::new(region, instruction.regions(), mappings)?;
            let device_outputs =
                self.driver.bind(self.context, instruction.operation().clone(), regions, &device_inputs)?;
            check_count!("output", device_outputs, outputs.len(), ProgramError);
            for (output, value) in outputs.iter_mut().zip(device_outputs) {
                output.push(value);
            }
        }
        Ok(outputs)
    }

    /// Returns `true` when the closure of the regions attached to `instruction` contains a collective over an emulated
    /// axis.
    fn uses_emulated_axes(
        &self,
        region: RegionRef<'_, ArrayIrValue<Array>, C::Operation>,
        instruction: &Instruction<C::Operation>,
    ) -> Result<bool, ProgramError> {
        for id in instruction.regions() {
            if region.with_id(*id)?.instructions_in_closure().any(|(_, nested)| {
                collective_axis_name(nested.operation()).is_some_and(|axis_name| self.grid.axis(axis_name).is_some())
            }) {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Computes `operation` across the devices when it is a collective over an emulated axis, and returns [`None`]
    /// otherwise. The trailing first-class dimension inputs of the mixed forms of the shape-changing collectives
    /// describe the result extents, which the emulator validates against the computed results.
    fn run_collective(
        &self,
        operation: &C::Operation,
        inputs: &[DeviceValues],
    ) -> Result<Option<Vec<DeviceValues>>, ProgramError> {
        let Some(axis) = collective_axis_name(operation).and_then(|axis_name| self.grid.axis(axis_name)) else {
            return Ok(None);
        };
        let name = operation.name();
        let (axis_name, axis_size) = &self.grid.axes[axis];
        let validate_axis_size = |payload_axis_size: usize| {
            if payload_axis_size == *axis_size {
                Ok(())
            } else {
                Err(ProgramError::MalformedProgram(format!(
                    "`{name}` was staged for {payload_axis_size} participants, but manual axis `{axis_name}` has \
                     {axis_size} devices",
                )))
            }
        };
        if let Some(payload) = operation.projected_payload::<AxisIndexOperation>() {
            check_count!("input", inputs, 0, ProgramError);
            let output_type = single_output_type(payload, &[])?;
            return Ok(Some(vec![self.grid.map_devices(|device| {
                let position = Array::scalar(self.grid.coordinates(device)[axis] as u64)?;
                retype(&position.converted_to(output_type.data_type())?, &output_type)
            })?]));
        }
        if let Some(payload) = operation.projected_payload::<ParallelRaggedAllToAllOperation>() {
            // The devices that differ only along the axis exchange segments in one host-side exchange, whose
            // participants are ordered by their coordinates and whose participant groups select the exchange partners.
            // Unlike the other collectives, the exchange runs through the operation's own physical representation,
            // which keeps one implementation of its offset, size, participant group, and update semantics.
            validate_axis_size(payload.axis_size())?;
            check_count!("input", inputs, 6, ProgramError);
            let mut outputs = vec![None; self.grid.device_count()];
            for device in (0..self.grid.device_count()).filter(|device| self.grid.coordinates(*device)[axis] == 0) {
                let (participants, _) = self.grid.participants(device, axis, None)?;
                let participant_inputs = participants
                    .iter()
                    .map(|participant| {
                        inputs
                            .iter()
                            .map(|values| array_value(&values[*participant], name).cloned())
                            .collect::<Result<Vec<_>, _>>()
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                // Validate each participant at its logical type before packing unplaced physical arrays. The
                // inferred output types retain its mesh and manual variation for restoration after the exchange.
                let output_types = participant_inputs
                    .iter()
                    .map(|inputs| {
                        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
                        Ok(payload.infer_output_types(&input_types, &[])?.remove(0))
                    })
                    .collect::<Result<Vec<_>, ProgramError>>()?;

                // Each of the six physical inputs has one leading row per participant, in axis-index order.
                // Corresponding inputs must have one static shape and data type to share that representation.
                let stacked_inputs = (0..6)
                    .map(|index| {
                        let input_type = participant_inputs[0][index].r#type();
                        let Some(shape) = input_type.static_shape() else {
                            return Err(TypeError::invalid(format!(
                                "`{name}` participant input {index} must have a static shape but got `{input_type}`",
                            ))
                            .into());
                        };
                        let mut stacked_shape = vec![participants.len()];
                        stacked_shape.extend(shape.dimensions());
                        let mut bytes = Vec::new();
                        for inputs in &participant_inputs {
                            let participant_type = inputs[index].r#type();
                            if participant_type.data_type() != input_type.data_type()
                                || participant_type.shape() != input_type.shape()
                            {
                                return Err(TypeError::invalid(format!(
                                    "`{name}` participant inputs {index} must share one static shape and data type but \
                                     got `{input_type}` and `{participant_type}`",
                                ))
                                .into());
                            }
                            bytes.extend(inputs[index].logical_bytes());
                        }
                        Array::from_logical_bytes(ArrayType::new_static(input_type.data_type(), stacked_shape), &bytes)
                    })
                    .collect::<Result<Vec<_>, ProgramError>>()?;
                let mut stacked_outputs = payload.with_physical_representation(None).interpret(
                    &EagerContext::<Array, ArrayOperation<Array>>::new(),
                    &EmptyRegionDriver,
                    &stacked_inputs,
                )?;
                check_count!("output", stacked_outputs, 1, ProgramError);

                // Split the physical result in the same participant order and restore each logical output type.
                let stacked_bytes = stacked_outputs.remove(0).logical_bytes();
                let row_byte_count = stacked_bytes.len() / participants.len();
                for (position, (participant, output_type)) in participants.iter().zip(output_types).enumerate() {
                    let row = position * row_byte_count..(position + 1) * row_byte_count;
                    let output = Array::from_logical_bytes(output_type, &stacked_bytes[row])?;
                    outputs[*participant] = Some(ArrayIrValue::Array(output));
                }
            }
            return Ok(Some(vec![outputs.into_iter().map(Option::unwrap).collect()]));
        }
        // Every remaining collective consumes one array per device, followed by the result extents of its mixed form.
        let (values, extents) =
            inputs.split_first().ok_or(ProgramError::InvalidInputCount { expected: 1, actual: 0 })?;
        let arrays = values.iter().map(|value| array_value(value, name)).collect::<Result<Vec<_>, _>>()?;
        let outputs = if let Some(payload) = operation.projected_payload::<ParallelVaryOperation>() {
            // Varying an invariant value along an axis changes only its type, because every device already holds it.
            self.grid.map_devices(|device| {
                retype(arrays[device], &collective_output_type(payload, arrays[device], extents, device)?)
            })?
        } else if let Some(payload) = operation.projected_payload::<ParallelReduceOperation>() {
            if let Some(payload_axis_size) = payload.axis_size() {
                validate_axis_size(payload_axis_size)?;
            }
            self.grid.map_devices(|device| {
                let (participants, _) = self.grid.participants(device, axis, payload.axis_index_groups())?;
                let participants = participants.iter().map(|participant| arrays[*participant]).collect::<Vec<_>>();
                retype(
                    &reduce_participants(&participants, payload.kind())?,
                    &collective_output_type(payload, arrays[device], extents, device)?,
                )
            })?
        } else if let Some(payload) = operation.projected_payload::<ParallelAllGatherOperation>() {
            validate_axis_size(payload.axis_size())?;
            self.grid.map_devices(|device| {
                let output_type = collective_output_type(payload, arrays[device], extents, device)?;
                let (participants, _) = self.grid.participants(device, axis, payload.options().axis_index_groups())?;
                let mut bytes = LogicalBuffer::zeros(&output_type)?;
                let axis = payload.concatenation_axis();
                for (position, participant) in participants.iter().enumerate() {
                    let input = arrays[*participant];
                    let mut shape = static_shape(&input.r#type())?;
                    let mut offsets = vec![0; shape.len()];
                    match payload.options().mode() {
                        CollectiveMode::Untiled => {
                            shape.insert(axis, 1);
                            offsets.insert(axis, position);
                        }
                        CollectiveMode::Tiled => offsets[axis] = position * shape[axis],
                    }
                    bytes.write_block(&offsets, &shape, &input.logical_bytes());
                }
                bytes.into_array(output_type)
            })?
        } else if let Some(payload) = operation.projected_payload::<ParallelSumScatterOperation>() {
            validate_axis_size(payload.axis_size())?;
            self.grid.map_devices(|device| {
                let output_type = collective_output_type(payload, arrays[device], extents, device)?;
                let groups = payload.options().axis_index_groups();
                let (participants, position) = self.grid.participants(device, axis, groups)?;
                let participant_count = participants.len();
                let participants = participants.iter().map(|participant| arrays[*participant]).collect::<Vec<_>>();
                let sum = LogicalBuffer::from_array(&reduce_participants(&participants, ReductionKind::Sum)?)?;
                let axis = payload.scatter_axis();
                let mut shape = sum.shape.clone();
                shape[axis] = match payload.options().mode() {
                    CollectiveMode::Untiled => 1,
                    CollectiveMode::Tiled => shape[axis] / participant_count,
                };
                let mut offsets = vec![0; shape.len()];
                offsets[axis] = position * shape[axis];
                Array::from_logical_bytes(output_type, &sum.read_block(&offsets, &shape))
            })?
        } else if let Some(payload) = operation.projected_payload::<ParallelAllToAllOperation>() {
            validate_axis_size(payload.axis_size())?;
            self.grid.map_devices(|device| {
                let output_type = collective_output_type(payload, arrays[device], extents, device)?;
                let groups = payload.options().axis_index_groups();
                let (participants, position) = self.grid.participants(device, axis, groups)?;
                let (split_axis, concatenation_axis) = (payload.split_axis(), payload.concatenation_axis());
                let mut bytes = LogicalBuffer::zeros(&output_type)?;
                for (sender_position, sender) in participants.iter().enumerate() {
                    // Every sender contributes the chunk of its split axis that belongs to this device's position.
                    let input = LogicalBuffer::from_array(arrays[*sender])?;
                    let mut chunk_shape = input.shape.clone();
                    chunk_shape[split_axis] = match payload.options().mode() {
                        CollectiveMode::Untiled => 1,
                        CollectiveMode::Tiled => chunk_shape[split_axis] / participants.len(),
                    };
                    let mut chunk_offsets = vec![0; chunk_shape.len()];
                    chunk_offsets[split_axis] = position * chunk_shape[split_axis];
                    let chunk = input.read_block(&chunk_offsets, &chunk_shape);

                    // The chunks are stacked along a new sender axis (untiled) or concatenated along an existing one
                    // (tiled) in the order of their senders' positions.
                    let mut offsets = vec![0; chunk_shape.len()];
                    match payload.options().mode() {
                        CollectiveMode::Untiled => {
                            chunk_shape.remove(split_axis);
                            chunk_shape.insert(concatenation_axis, 1);
                            offsets[concatenation_axis] = sender_position;
                        }
                        CollectiveMode::Tiled => {
                            offsets[concatenation_axis] = sender_position * chunk_shape[concatenation_axis];
                        }
                    }
                    bytes.write_block(&offsets, &chunk_shape, &chunk);
                }
                bytes.into_array(output_type)
            })?
        } else {
            // `parallel_permute` is the only remaining collective that `collective_axis_name` recognizes.
            let payload = operation.projected_payload::<ParallelPermuteOperation>().unwrap();
            validate_axis_size(payload.axis_size())?;
            self.grid.map_devices(|device| {
                // A device that no pair targets receives zeros.
                let output_type = collective_output_type(payload, arrays[device], extents, device)?;
                let coordinates = self.grid.coordinates(device);
                match payload.source_target_pairs().iter().find(|(_, target)| *target == coordinates[axis]) {
                    Some((source, _)) => {
                        let mut source_coordinates = coordinates.clone();
                        source_coordinates[axis] = *source;
                        retype(arrays[self.grid.device(&source_coordinates)], &output_type)
                    }
                    None => EagerContext::<Array, ArrayOperation<Array>>::new().zero(&output_type),
                }
            })?
        };
        Ok(Some(vec![outputs]))
    }

    /// Runs a `while` loop whose condition or body uses collectives over the emulated axes in lockstep, applying the
    /// same masked state update as the eager rule on every device.
    fn run_while(
        &self,
        operation: &WhileOperation<ArrayIrType>,
        condition: RegionRef<'_, ArrayIrValue<Array>, C::Operation>,
        body: RegionRef<'_, ArrayIrValue<Array>, C::Operation>,
        inputs: &[DeviceValues],
    ) -> Result<Vec<DeviceValues>, ProgramError> {
        let mut state = inputs.to_vec();
        let mut completed_iterations = 0;
        loop {
            if operation.iteration_bound().is_some_and(|bound| completed_iterations >= bound) {
                return Ok(state);
            }
            let mut predicates = self.run_region(condition, state.clone())?;
            check_count!("output", predicates, 1, ProgramError);
            let predicates = predicates.remove(0);
            let continuing = predicates.iter().map(WhilePredicate::any_true).collect::<Result<Vec<_>, _>>()?;
            if continuing.iter().all(|continuing| !continuing) {
                return Ok(state);
            }
            if !continuing.iter().all(|continuing| *continuing) {
                return Err(divergent_predicates_error(operation.name()));
            }
            let candidates = self.run_region(body, state.clone())?;
            check_count!("output", candidates, state.len(), ProgramError);
            state = candidates
                .iter()
                .zip(&state)
                .map(|(candidates, carried)| {
                    predicates
                        .iter()
                        .zip(candidates.iter().zip(carried))
                        .map(|(predicate, (candidate, carried))| predicate.mask_select(candidate, carried))
                        .collect()
                })
                .collect::<Result<Vec<_>, ProgramError>>()?;
            completed_iterations += 1;
        }
    }

    /// Runs the branch of a `condition` whose branches use collectives over the emulated axes in lockstep, which
    /// requires every device to select the same branch.
    fn run_condition(
        &self,
        branches: &[RegionRef<'_, ArrayIrValue<Array>, C::Operation>],
        inputs: &[DeviceValues],
    ) -> Result<Vec<DeviceValues>, ProgramError> {
        let (predicates, inputs) =
            inputs.split_first().ok_or(ProgramError::InvalidInputCount { expected: 1, actual: 0 })?;
        let predicates = predicates.iter().map(Concretizable::<bool>::concretize).collect::<Result<Vec<_>, _>>()?;
        if predicates.iter().any(|predicate| *predicate != predicates[0]) {
            return Err(divergent_predicates_error(CONDITION_OPERATION_NAME));
        }
        self.run_region(branches[if predicates[0] { 0 } else { 1 }], inputs.to_vec())
    }

    /// Runs a `scan` whose body uses collectives over the emulated axes in lockstep, following the eager rule for
    /// static trip counts: every iteration receives the slice of each array stack, while reference carries and stacks
    /// pass through as the roots that the body indexes itself.
    fn run_scan(
        &self,
        operation: &ScanOperation<ArrayIrType>,
        body: RegionRef<'_, ArrayIrValue<Array>, C::Operation>,
        inputs: &[DeviceValues],
    ) -> Result<Vec<DeviceValues>, ProgramError> {
        let name = operation.name();
        let &Dimension::Static(length) = operation.length() else {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "`{SHARD_MAP_OPERATION_NAME}` interpretation cannot run `{name}` with a dynamic length in \
                     lockstep across devices",
                ),
            });
        };
        let carry_count = operation.carry_count();
        check_count!("input", inputs, body.input_ids().len().saturating_sub(1), ProgramError);
        let device_count = self.grid.device_count();
        let zero_context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let mut accumulators = body.output_types()[carry_count..]
            .iter()
            .map(|slice_type| {
                let stacked_type = stacked_scan_type(<&ArrayType>::try_from(slice_type)?, length);
                (0..device_count).map(|_| zero_context.zero(&stacked_type)).collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, ProgramError>>()?;
        let (carries, stacks) = inputs.split_at(carry_count);
        let mut carries = carries.to_vec();
        for visit in 0..length {
            let iteration = if operation.reverse() { length - 1 - visit } else { visit };
            let mut iteration_inputs = vec![vec![ArrayIrValue::Array(Array::scalar(iteration as i64)?); device_count]];
            iteration_inputs.extend(carries.iter().cloned());
            for stack in stacks {
                iteration_inputs.push(
                    stack
                        .iter()
                        .map(|value| match value {
                            ArrayIrValue::Reference(_) => Ok(value.clone()),
                            _ => Ok(ArrayIrValue::Array(read_scan_iteration(array_value(value, name)?, iteration)?)),
                        })
                        .collect::<Result<Vec<_>, ProgramError>>()?,
                );
            }
            let mut outputs = self.run_region(body, iteration_inputs)?;
            check_count!("output", outputs, carry_count + accumulators.len(), ProgramError);
            let stacked_outputs = outputs.split_off(carry_count);
            for (position, (carried, next)) in carries.iter().zip(&outputs).enumerate() {
                for (carried, next) in carried.iter().zip(next) {
                    if let ArrayIrValue::Reference(carried) = carried
                        && !matches!(next, ArrayIrValue::Reference(next) if next.id() == carried.id())
                    {
                        return Err(ProgramError::MalformedProgram(format!(
                            "operation `{SCAN_OPERATION_NAME}` does not return carry {position} as the reference it \
                             entered with, so its `{SCAN_OPERATION_NAME}` state has no fixed point",
                        )));
                    }
                }
            }
            carries = outputs;
            for (accumulators, outputs) in accumulators.iter_mut().zip(stacked_outputs) {
                for (accumulator, output) in accumulators.iter_mut().zip(outputs) {
                    let output = array_value(&output, name)?.clone();
                    *accumulator = write_scan_iteration(accumulator.clone(), iteration, output)?;
                }
            }
        }
        carries.extend(
            accumulators
                .into_iter()
                .map(|accumulators| accumulators.into_iter().map(ArrayIrValue::Array).collect()),
        );
        Ok(carries)
    }
}

/// Returns the name of the axis that `operation` communicates along when it is a collective (including `axis_index`,
/// which observes the coordinate along that axis), and [`None`] otherwise.
pub(super) fn collective_axis_name<O: OperationPayloadProjection>(operation: &O) -> Option<&str> {
    operation
        .projected_payload::<AxisIndexOperation>()
        .map(AxisIndexOperation::axis_name)
        .or_else(|| operation.projected_payload::<ParallelReduceOperation>().map(ParallelReduceOperation::axis_name))
        .or_else(|| operation.projected_payload::<ParallelVaryOperation>().map(ParallelVaryOperation::axis_name))
        .or_else(|| {
            operation
                .projected_payload::<ParallelAllGatherOperation>()
                .map(ParallelAllGatherOperation::axis_name)
        })
        .or_else(|| {
            operation
                .projected_payload::<ParallelSumScatterOperation>()
                .map(ParallelSumScatterOperation::axis_name)
        })
        .or_else(|| {
            operation.projected_payload::<ParallelAllToAllOperation>().map(ParallelAllToAllOperation::axis_name)
        })
        .or_else(|| operation.projected_payload::<ParallelPermuteOperation>().map(ParallelPermuteOperation::axis_name))
        .or_else(|| {
            operation
                .projected_payload::<ParallelRaggedAllToAllOperation>()
                .map(ParallelRaggedAllToAllOperation::axis_name)
        })
}

/// Returns the error reported when devices take different paths through a region operation that runs in lockstep.
fn divergent_predicates_error(operation_name: &str) -> ProgramError {
    ProgramError::UnsupportedOperation {
        message: format!(
            "the devices of a `{SHARD_MAP_OPERATION_NAME}` disagree on the predicate of a `{operation_name}` whose \
             regions use collectives over the manual axes, so the collectives would not have matching participants",
        ),
    }
}

/// Returns the array that `value` holds, rejecting dimensions and references, which no boundary or collective of
/// `operation_name` accepts.
fn array_value<'v>(value: &'v ArrayIrValue<Array>, operation_name: &str) -> Result<&'v Array, ProgramError> {
    match value {
        ArrayIrValue::Array(array) => Ok(array),
        _ => Err(ProgramError::InvalidArgument {
            message: format!("`{operation_name}` expects an array value but got `{}`", value.r#type()),
        }),
    }
}

/// Returns the single output type that `operation` infers for `input_types`.
fn single_output_type<O: Operation<Type = ArrayType>>(
    operation: &O,
    input_types: &[ArrayType],
) -> Result<ArrayType, ProgramError> {
    let mut output_types = operation.infer_output_types(input_types, &[])?;
    check_count!("output", output_types, 1, ProgramError);
    Ok(output_types.remove(0))
}

/// Returns the output type of the collective `payload` on `device`, whose array input is `input`, after validating that
/// the trailing first-class dimension inputs of its mixed form, if any, are exactly the extents of that output type.
fn collective_output_type<P: Operation<Type = ArrayType>>(
    payload: &P,
    input: &Array,
    extents: &[DeviceValues],
    device: usize,
) -> Result<ArrayType, ProgramError> {
    let output_type = single_output_type(payload, &[input.r#type().into_owned()])?;
    if extents.is_empty() {
        return Ok(output_type);
    }
    let shape = static_shape(&output_type)?;
    if extents.len() != shape.len() {
        return Err(ProgramError::InvalidInputCount { expected: shape.len() + 1, actual: extents.len() + 1 });
    }
    for (axis, (extent, expected)) in extents.iter().map(|extents| &extents[device]).zip(shape).enumerate() {
        match extent {
            ArrayIrValue::Dimension(extent) if extent.extent() == expected => {}
            _ => {
                return Err(ProgramError::InvalidArgument {
                    message: format!(
                        "`{}` result extent input #{axis} is `{extent}`, but result axis #{axis} has extent \
                         {expected}",
                        payload.name(),
                    ),
                });
            }
        }
    }
    Ok(output_type)
}

/// Returns the static dimensions of `array_type`.
fn static_shape(array_type: &ArrayType) -> Result<Vec<usize>, ProgramError> {
    array_type
        .static_shape()
        .map(|shape| shape.dimensions().to_vec())
        .ok_or_else(|| ProgramError::InvalidArgument {
            message: format!(
                "`{SHARD_MAP_OPERATION_NAME}` interpretation requires static types but got `{array_type}`"
            ),
        })
}

/// Returns `value` with the same logical elements but of type `type`, which must have the data type and the element
/// count of the type of `value`.
fn retype(value: &Array, r#type: &ArrayType) -> Result<Array, ProgramError> {
    if value.r#type().data_type() != r#type.data_type() {
        return Err(ProgramError::MalformedProgram(format!(
            "cannot retype a value of type `{}` to type `{type}`",
            value.r#type(),
        )));
    }
    Array::from_logical_bytes(r#type.clone(), &value.logical_bytes())
}

/// Reduces the values of the participants of a collective elementwise with `kind`, through the same kernel as a
/// reduction across an array axis, and returns the result at a plain (unplaced, dense) type with the participants'
/// shape and data type.
fn reduce_participants(participants: &[&Array], kind: ReductionKind) -> Result<Array, ProgramError> {
    let participant_type = participants[0].r#type();
    let data_type = participant_type.data_type();
    let mut stacked_shape = vec![participants.len()];
    stacked_shape.extend(static_shape(&participant_type)?);
    let stacked_bytes = participants.iter().flat_map(|participant| participant.logical_bytes()).collect::<Vec<_>>();
    let stacked = Array::from_logical_bytes(ArrayType::new_static(data_type, stacked_shape), &stacked_bytes)?;
    let reduced = stacked.reduce(&[0], kind)?;
    if reduced.r#type().data_type() == data_type { Ok(reduced) } else { reduced.converted_to(data_type) }
}

/// Returns the offset of the block that the device at `coordinates` holds, along every dimension of a boundary value
/// placed by `sharding`. Along a dimension sharded over active manual axes (which precede every free axis of that
/// dimension), the offset is the row-major partition index over those axes scaled by the local extent `local_shape`
/// of the dimension. Free axes do not partition the local value, so they do not contribute.
fn manual_block_offsets(
    sharding: &Sharding,
    grid: &ManualDeviceGrid,
    coordinates: &[usize],
    local_shape: &[usize],
) -> Vec<usize> {
    sharding
        .dimensions()
        .iter()
        .zip(local_shape)
        .map(|(dimension, extent)| match dimension {
            ShardingDimension::Sharded(axis_names) => {
                let partition = axis_names.iter().fold(0, |partition, axis_name| match grid.axis(axis_name) {
                    Some(axis) => partition * grid.axes[axis].1 + coordinates[axis],
                    None => partition,
                });
                partition * extent
            }
            ShardingDimension::Replicated | ShardingDimension::Unconstrained => 0,
        })
        .collect()
}

/// Returns the shard of global input `input_index` that the device at `coordinates` of the local `grid` (i.e., of the
/// active manual axes) receives, at type `local_type`. A device whose coordinate along an unreduced active manual axis
/// is non-zero receives zeros, so that the shards sum to the logical value that the global [`Array`] stores.
fn split_input(
    shard_map: &ShardMap,
    input_index: usize,
    value: &Array,
    local_type: &ArrayType,
    grid: &ManualDeviceGrid,
    coordinates: &[usize],
) -> Result<Array, ProgramError> {
    let sharding = &shard_map.in_shardings()[input_index];
    if sharding
        .unreduced_axes()
        .iter()
        .any(|axis_name| grid.axis(axis_name).is_some_and(|axis| coordinates[axis] != 0))
    {
        return EagerContext::<Array, ArrayOperation<Array>>::new().zero(local_type);
    }
    let local_shape = static_shape(local_type)?;
    let offsets = manual_block_offsets(sharding, grid, coordinates, &local_shape);
    let shard = LogicalBuffer::from_array(value)?.read_block(&offsets, &local_shape);
    Array::from_logical_bytes(local_type.clone(), &shard)
}

/// Assembles a global boundary value of type `global_type` (i.e., a global output or the referent of a reference
/// input) from the local `values` of the devices of `grid` (i.e., of the local grid of one enclosing device), which
/// `sharding` places. Every device places its value at its tile of `sharding`. Along an active manual axis that
/// `sharding` does not tile, the devices hold partial summands when the axis is unreduced, which are summed, and
/// otherwise hold the same value, of which the device at coordinate zero contributes.
fn assemble(
    sharding: &Sharding,
    global_type: &ArrayType,
    grid: &ManualDeviceGrid,
    values: &[ArrayIrValue<Array>],
) -> Result<Array, ProgramError> {
    let tiled_axes = tiled_grid_axes(sharding, grid);
    let unreduced_axes =
        sharding.unreduced_axes().iter().filter_map(|axis_name| grid.axis(axis_name)).collect::<Vec<_>>();
    let mut tiles = Vec::<(Vec<usize>, Vec<&Array>)>::new();
    let mut local_shape = Vec::new();
    for (device, value) in values.iter().enumerate() {
        let coordinates = grid.coordinates(device);
        let contributes = coordinates
            .iter()
            .enumerate()
            .all(|(axis, coordinate)| *coordinate == 0 || tiled_axes.contains(&axis) || unreduced_axes.contains(&axis));
        if !contributes {
            continue;
        }
        let value = array_value(value, SHARD_MAP_OPERATION_NAME)?;
        local_shape = static_shape(&value.r#type())?;
        let offsets = manual_block_offsets(sharding, grid, &coordinates, &local_shape);
        match tiles.iter_mut().find(|(tile_offsets, _)| *tile_offsets == offsets) {
            Some((_, summands)) => summands.push(value),
            None => tiles.push((offsets, vec![value])),
        }
    }
    let mut output = LogicalBuffer::zeros(global_type)?;
    for (offsets, summands) in tiles {
        let tile = if summands.len() == 1 {
            summands[0].logical_bytes()
        } else {
            reduce_participants(&summands, ReductionKind::Sum)?.logical_bytes()
        };
        output.write_block(&offsets, &local_shape, &tile);
    }
    output.into_array(global_type.clone())
}

/// Returns the positions of the axes of `grid` along which `sharding` tiles a value.
fn tiled_grid_axes(sharding: &Sharding, grid: &ManualDeviceGrid) -> Vec<usize> {
    sharding
        .dimensions()
        .iter()
        .flat_map(|dimension| match dimension {
            ShardingDimension::Sharded(axis_names) => axis_names.as_slice(),
            ShardingDimension::Replicated | ShardingDimension::Unconstrained => &[],
        })
        .filter_map(|axis_name| grid.axis(axis_name))
        .collect()
}

/// Validates that the final local `values` of reference input `input_index`, one per device of `grid`, which
/// `sharding` places, are identical copies along every axis of `grid` that `sharding` neither tiles nor marks as
/// unreduced. Each device holds its own copy of the referent along such an axis, and the reference contract of
/// `shard_map` admits only mutations that keep the copies identical, so copies that differ indicate a body that
/// violates it (e.g., one that writes a reference from a branch that the devices select differently) rather than
/// values that could be assembled. Along an unreduced axis, the devices hold partial states (e.g., the logical value
/// on the device at coordinate zero and zeros elsewhere, as split), which [`assemble`] sums instead.
///
/// # Errors
///
/// Returns [`ProgramError::MalformedProgram`] naming the first axis along which two copies differ.
fn validate_replicated_copies(
    sharding: &Sharding,
    grid: &ManualDeviceGrid,
    values: &[ArrayIrValue<Array>],
    input_index: usize,
) -> Result<(), ProgramError> {
    let tiled_axes = tiled_grid_axes(sharding, grid);
    let unreduced_axes =
        sharding.unreduced_axes().iter().filter_map(|axis_name| grid.axis(axis_name)).collect::<Vec<_>>();
    for (device, value) in values.iter().enumerate() {
        let coordinates = grid.coordinates(device);
        for (axis, coordinate) in coordinates.iter().enumerate() {
            if *coordinate == 0 || tiled_axes.contains(&axis) || unreduced_axes.contains(&axis) {
                continue;
            }
            let mut first_coordinates = coordinates.clone();
            first_coordinates[axis] = 0;
            let first = array_value(&values[grid.device(&first_coordinates)], SHARD_MAP_OPERATION_NAME)?;
            if array_value(value, SHARD_MAP_OPERATION_NAME)?.logical_bytes() != first.logical_bytes() {
                return Err(ProgramError::MalformedProgram(format!(
                    "`{SHARD_MAP_OPERATION_NAME}` reference input #{input_index} is replicated along manual axis `{}`, \
                     but its devices along that axis finished with different states",
                    grid.axes[axis].0,
                )));
            }
        }
    }
    Ok(())
}

/// Row-major logical element encodings of a static array (i.e., the representation of [`Array::logical_bytes`]),
/// together with the shape and element width needed to address rectangular blocks of them.
struct LogicalBuffer {
    /// Logical element encodings in row-major order.
    bytes: Vec<u8>,

    /// Static shape of the array.
    shape: Vec<usize>,

    /// Number of bytes of each logical element encoding.
    element_width: usize,
}

impl LogicalBuffer {
    /// Creates a new [`LogicalBuffer`] holding the logical elements of `array`.
    fn from_array(array: &Array) -> Result<Self, ProgramError> {
        let array_type = array.r#type();
        Ok(Self {
            bytes: array.logical_bytes(),
            shape: static_shape(&array_type)?,
            element_width: ArrayAddressing::element_byte_width_for_data_type(array_type.data_type()),
        })
    }

    /// Creates a new [`LogicalBuffer`] of type `array_type` whose element encodings are all zero bytes. Every block
    /// of it must be written before [`Self::into_array`] is called, because zero bytes need not encode zero.
    fn zeros(array_type: &ArrayType) -> Result<Self, ProgramError> {
        let shape = static_shape(array_type)?;
        let element_width = ArrayAddressing::element_byte_width_for_data_type(array_type.data_type());
        Ok(Self { bytes: vec![0; shape.iter().product::<usize>() * element_width], shape, element_width })
    }

    /// Returns the flat element index of the element at `offsets + index`.
    fn flat_index(&self, offsets: &[usize], index: &[usize]) -> usize {
        offsets
            .iter()
            .zip(index)
            .zip(&self.shape)
            .fold(0, |flat, ((offset, index), size)| flat * size + offset + index)
    }

    /// Returns the logical element encodings of the block of shape `block_shape` that starts at `offsets`.
    fn read_block(&self, offsets: &[usize], block_shape: &[usize]) -> Vec<u8> {
        let mut block = Vec::with_capacity(block_shape.iter().product::<usize>() * self.element_width);
        for_each_index(block_shape, |index| {
            let start = self.flat_index(offsets, index) * self.element_width;
            block.extend_from_slice(&self.bytes[start..start + self.element_width]);
        });
        block
    }

    /// Overwrites the block of shape `block_shape` that starts at `offsets` with the logical element encodings
    /// `block`.
    fn write_block(&mut self, offsets: &[usize], block_shape: &[usize], block: &[u8]) {
        let mut element = 0;
        for_each_index(block_shape, |index| {
            let start = self.flat_index(offsets, index) * self.element_width;
            self.bytes[start..start + self.element_width]
                .copy_from_slice(&block[element * self.element_width..(element + 1) * self.element_width]);
            element += 1;
        });
    }

    /// Returns the [`Array`] of type `array_type` whose logical elements are those of this buffer.
    fn into_array(self, array_type: ArrayType) -> Result<Array, ProgramError> {
        Array::from_logical_bytes(array_type, &self.bytes)
    }
}

/// Calls `function` with every index of an array of shape `shape`, in row-major order.
fn for_each_index<F: FnMut(&[usize])>(shape: &[usize], mut function: F) {
    if shape.contains(&0) {
        return;
    }
    let mut index = vec![0; shape.len()];
    loop {
        function(&index);
        let Some(axis) = (0..shape.len()).rev().find(|axis| index[*axis] + 1 < shape[*axis]) else {
            return;
        };
        index[axis] += 1;
        index[axis + 1..].fill(0);
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::sync::Arc;

    use pretty_assertions::assert_eq;

    use crate::arrays::{
        ArrayIrOperation, ArrayReferenceTransform, DataType, DimensionBounds, DimensionVariable, LogicalMesh, MeshAxis,
        MeshAxisType, Shape,
    };
    use crate::contexts::Context;
    use crate::differentiation::NothingSavable;
    use crate::operations::arithmetic::AddOperation;
    use crate::operations::collectives::CollectiveOptions;
    use crate::operations::collectives::axis_index::AxisIndex;
    use crate::operations::collectives::parallel_all_gather::ParallelAllGather;
    use crate::operations::collectives::parallel_all_to_all::ParallelAllToAll;
    use crate::operations::collectives::parallel_permute::ParallelPermute;
    use crate::operations::collectives::parallel_ragged_all_to_all::ParallelRaggedAllToAll;
    use crate::operations::collectives::parallel_reduce::ParallelReduce;
    use crate::operations::collectives::parallel_sum_scatter::ParallelSumScatter;
    use crate::operations::collectives::parallel_vary::ParallelVary;
    use crate::operations::comparisons::{CompareOperation, ComparisonDirection};
    use crate::operations::constants::zero_like::ZeroLikeOperation;
    use crate::operations::custom_functions::operations::{
        CustomFunctionJvpRule, CustomFunctionOperation, CustomFunctionTransposeOperation,
    };
    use crate::operations::debugging::PrintOperation;
    use crate::operations::differentiation::linear_call::LinearCallOperation;
    use crate::operations::differentiation::rematerialize::RematerializeOperation;
    use crate::operations::dimensions::dimension_from_scalar::DimensionFromScalarOperation;
    use crate::operations::manipulation::broadcasting::DynamicBroadcastOperation;
    use crate::operations::manipulation::conversions::ConvertElementTypeOperation;
    use crate::operations::manipulation::reshaping::{Reshape, ReshapeOperation};
    use crate::operations::reductions::ReduceOperation;
    use crate::operations::references::{ReferenceAddUpdateOperation, ReferenceReadOperation, ReferenceWriteOperation};
    use crate::operations::sharding::shard_map::{ShardMapContext, ShardMapTracer, shard_map, shard_map_in_context};
    use crate::parameters::Placeholder;
    use crate::partial::ResidualPolicyReference;
    use crate::programs::{
        AtomId, BindingRegionDriver, EffectClass, Program, ProgramBuilder, ProjectedValue, ReferenceType, RegionDriver,
        ValueProjection,
    };
    use crate::tracing::{Tracer, TracingContext};

    use super::*;

    type TestValue = ArrayIrValue<Array>;
    type TestOperation = ArrayIrOperation<Array>;
    type TestProgram = Program<TestValue, TestOperation, Vec<TestValue>, Vec<TestValue>>;
    type TestContext = TracingContext<TestValue, TestOperation>;
    type TestTracer = ShardMapTracer<TestContext>;

    /// Two-device manual mesh over `x`.
    fn manual_mesh() -> LogicalMesh {
        LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap()
    }

    /// Four-device manual mesh over `x` and `y`.
    fn manual_mesh_2x2() -> LogicalMesh {
        LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap()
    }

    /// Sharding over `mesh` with the provided per-dimension assignments.
    fn sharding(mesh: &LogicalMesh, dimensions: Vec<ShardingDimension>) -> Sharding {
        Sharding::new(mesh.clone(), dimensions).unwrap()
    }

    /// Static `f32` array type of the provided shape, placed by `sharding` when one is provided.
    fn f32_type(shape: &[usize], sharding: Option<Sharding>) -> ArrayType {
        ArrayType::new_static(DataType::F32, shape.to_vec()).with_sharding(sharding).unwrap()
    }

    /// Composite `f32` array value of `type` with the provided logical elements.
    fn f32_value(r#type: ArrayType, elements: &[f32]) -> TestValue {
        ArrayIrValue::Array(Array::from_elements(r#type, elements).unwrap())
    }

    /// Traces a program over inputs of `input_types` whose outputs are the outputs of `function` over the array
    /// projections of its inputs.
    fn trace_program(
        function: impl FnOnce(Vec<TestTracer>) -> Vec<TestTracer>,
        input_types: Vec<ArrayType>,
    ) -> TestProgram {
        TestContext::trace_with_named_axes(
            |inputs: Vec<Tracer<TestContext>>| {
                let inputs = inputs
                    .into_iter()
                    .map(|input| ValueProjection::<ArrayType>::into_projected(input).map_err(ProgramError::from))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(function(inputs).into_iter().map(ProjectedValue::into_value).collect::<Vec<_>>())
            },
            input_types.into_iter().map(ArrayIrType::Array).collect::<Vec<_>>(),
            Vec::new(),
        )
        .unwrap()
        .1
    }

    /// Builds a program over inputs of `input_types` whose outputs are the atoms that `build` adds to it.
    fn build_program(
        input_types: Vec<ArrayIrType>,
        build: impl FnOnce(&mut ProgramBuilder<TestValue, TestOperation>, &[AtomId]) -> Vec<AtomId>,
    ) -> TestProgram {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let inputs = input_types.into_iter().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
        let outputs = build(&mut builder, &inputs);
        let output_count = outputs.len();
        builder.build(outputs, vec![Placeholder; inputs.len()], vec![Placeholder; output_count]).unwrap()
    }

    /// Stages the `shard_map` that `body` and `shard_map` define for `global_input_types` as the single instruction of
    /// a program whose inputs carry those global input types.
    fn shard_map_program(body: TestProgram, global_input_types: Vec<ArrayIrType>, shard_map: ShardMap) -> TestProgram {
        let operation = ShardMapOperation::from_program(&body, global_input_types.clone(), shard_map).unwrap();
        build_program(global_input_types, |builder, inputs| {
            let body = builder.import_program(body);
            builder.add_instruction(operation, vec![body], inputs.to_vec(), None).unwrap().to_vec()
        })
    }

    /// Returns the reference read operation of the array IR family.
    fn reference_read() -> ReferenceReadOperation<ArrayType, ArrayIrType, ArrayReferenceTransform> {
        ReferenceReadOperation::new()
    }

    /// Adds `parallel_vary(parallel_reduce(input))` over the manual axis `x` of `mesh` to `builder`, which sums `input`
    /// across the devices and keeps the sum varying along `x`, so that it can replace a value that varies along `x`.
    fn add_sum_over_x(
        builder: &mut ProgramBuilder<TestValue, TestOperation>,
        mesh: &LogicalMesh,
        input: AtomId,
    ) -> AtomId {
        let sum = ParallelReduceOperation::new(ReductionKind::Sum, "x".to_string()).with_mesh(mesh.clone());
        let sum =
            builder.add_instruction(ArrayOperation::ParallelReduce(sum), Vec::new(), vec![input], None).unwrap()[0];
        let vary = ArrayOperation::ParallelVary(ParallelVaryOperation::new("x".to_string()));
        builder.add_instruction(vary, Vec::new(), vec![sum], None).unwrap()[0]
    }

    #[test]
    fn test_shard_map_interpretation() {
        // Each device receives its tile of the input, and the output assembles the local outputs in mesh order.
        let mesh = manual_mesh();
        let sharded = sharding(&mesh, vec![ShardingDimension::sharded(["x"])]);
        let input = f32_value(f32_type(&[4], None), &[1.0, 2.0, 3.0, 4.0]);
        let identity = trace_program(
            |inputs| {
                vec![
                    shard_map(|x: TestTracer| x, inputs[0].clone(), mesh.clone(), sharded.clone(), sharded.clone())
                        .unwrap(),
                ]
            },
            vec![f32_type(&[4], None)],
        );
        assert_eq!(
            identity.interpret(vec![input.clone()]),
            Ok(vec![f32_value(f32_type(&[4], Some(sharded.clone())), &[1.0, 2.0, 3.0, 4.0])]),
        );

        // Ordinary operations of the body run on every device over that device's tile.
        let double = trace_program(
            |inputs| {
                vec![
                    shard_map(
                        |x: TestTracer| x.clone() + x,
                        inputs[0].clone(),
                        mesh.clone(),
                        sharded.clone(),
                        sharded.clone(),
                    )
                    .unwrap(),
                ]
            },
            vec![f32_type(&[4], None)],
        );
        assert_eq!(
            double.interpret(vec![input]),
            Ok(vec![f32_value(f32_type(&[4], Some(sharded.clone())), &[2.0, 4.0, 6.0, 8.0])]),
        );

        // A direct invocation without the attached body has nothing to replay.
        let operation = ShardMapOperation::from_boundary(
            ShardMap::from_shardings(mesh.clone(), vec![sharded.clone()], vec![sharded.clone()], vec!["x".to_string()]),
            vec![f32_type(&[4], Some(sharded.clone()))],
            vec![f32_type(&[4], Some(sharded))],
        );
        assert_eq!(
            operation.interpret(
                &EagerContext::<TestValue, TestOperation>::new(),
                &EmptyRegionDriver,
                &[f32_value(f32_type(&[4], None), &[1.0, 2.0, 3.0, 4.0])],
            ),
            Err(ProgramError::MalformedProgram(
                "operation `shard_map` declares 1 region slots but 0 regions were attached".to_string(),
            )),
        );
    }

    #[test]
    fn test_shard_map_interpretation_parallel_reduce() {
        // A reduction over the manual axis combines the tiles of both devices, so its invariant result is the output.
        let mesh = manual_mesh();
        let sharded = sharding(&mesh, vec![ShardingDimension::sharded(["x"])]);
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let reduce = |kind| {
            trace_program(
                |inputs| {
                    vec![
                        shard_map(
                            |x: TestTracer| x.parallel_reduce(kind, "x").unwrap(),
                            inputs[0].clone(),
                            mesh.clone(),
                            sharded.clone(),
                            replicated.clone(),
                        )
                        .unwrap(),
                    ]
                },
                vec![f32_type(&[4], None)],
            )
        };
        let input = f32_value(f32_type(&[4], None), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(
            reduce(ReductionKind::Sum).interpret(vec![input.clone()]),
            Ok(vec![f32_value(f32_type(&[2], Some(replicated.clone())), &[4.0, 6.0])]),
        );
        assert_eq!(
            reduce(ReductionKind::Max).interpret(vec![input]),
            Ok(vec![f32_value(f32_type(&[2], Some(replicated)), &[3.0, 4.0])]),
        );
    }

    #[test]
    fn test_shard_map_interpretation_axis_index_groups() {
        // A grouped reduction combines the values of the members of each participant group only, so the devices at
        // coordinates `0` and `1` receive the sum of their tiles, and the devices at coordinates `2` and `3` receive
        // the sum of theirs.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharded = sharding(&mesh, vec![ShardingDimension::sharded(["x"])]);
        let program = trace_program(
            |inputs| {
                vec![
                    shard_map(
                        |x: TestTracer| {
                            x.parallel_reduce_with_axis_index_groups(
                                ReductionKind::Sum,
                                "x",
                                vec![vec![0, 1], vec![2, 3]],
                            )
                            .unwrap()
                        },
                        inputs[0].clone(),
                        mesh.clone(),
                        sharded.clone(),
                        sharded.clone(),
                    )
                    .unwrap(),
                ]
            },
            vec![f32_type(&[4], None)],
        );
        assert_eq!(
            program.interpret(vec![f32_value(f32_type(&[4], None), &[1.0, 2.0, 3.0, 4.0])]),
            Ok(vec![f32_value(f32_type(&[4], Some(sharded)), &[3.0, 3.0, 7.0, 7.0])]),
        );

        // The participants of a device are the members of its group, positioned by their order in that group, and
        // groups that leave a coordinate without a group do not partition the axis.
        let grid = ManualDeviceGrid { axes: vec![("x".to_string(), 4)] };
        assert_eq!(grid.participants(2, 0, Some([vec![3, 2], vec![1, 0]].as_slice())), Ok((vec![3, 2], 1)));
        assert_eq!(
            grid.participants(2, 0, Some([vec![0, 1]].as_slice())),
            Err(ProgramError::InvalidArgument {
                message: "the participant groups `[[0, 1]]` of a collective over manual axis `x` do not partition \
                          its 4 coordinates"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_shard_map_interpretation_axis_index() {
        // Each device reads its own coordinate, and the tiled output assembles the coordinates of both devices.
        let mesh = manual_mesh();
        let sharded = sharding(&mesh, vec![ShardingDimension::sharded(["x"])]);
        let context = TestContext::new();
        let output = shard_map_in_context(
            &context,
            |context: &ShardMapContext<TestContext>, ()| context.axis_index("x").unwrap().reshape([1]).unwrap(),
            (),
            mesh,
            (),
            sharded.clone(),
            Vec::new(),
        )
        .unwrap();
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![output.value().atom_id().unwrap()],
                Vec::new(),
                vec![Placeholder],
            )
            .unwrap();
        let output_type = ArrayType::new_static(DataType::U64, [2]).with_sharding(sharded).unwrap();
        assert_eq!(
            program.interpret(Vec::new()),
            Ok(vec![ArrayIrValue::Array(Array::from_elements(output_type, &[0u64, 1]).unwrap())]),
        );
    }

    #[test]
    fn test_shard_map_interpretation_parallel_vary() {
        // Varying a replicated input along the manual axis retypes the copy that every device holds, so the tiled
        // output holds one copy per device.
        let mesh = manual_mesh();
        let sharded = sharding(&mesh, vec![ShardingDimension::sharded(["x"])]);
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let program = trace_program(
            |inputs| {
                vec![
                    shard_map(
                        |x: TestTracer| x.parallel_vary("x").unwrap(),
                        inputs[0].clone(),
                        mesh.clone(),
                        replicated.clone(),
                        sharded.clone(),
                    )
                    .unwrap(),
                ]
            },
            vec![f32_type(&[2], None)],
        );
        assert_eq!(
            program.interpret(vec![f32_value(f32_type(&[2], None), &[5.0, 6.0])]),
            Ok(vec![f32_value(f32_type(&[4], Some(sharded)), &[5.0, 6.0, 5.0, 6.0])]),
        );
    }

    #[test]
    fn test_shard_map_interpretation_partial_manual_mesh() {
        // Only the manual axis `x` is emulated, while the placement over the auto axis `y` stays with the values. The
        // rows of the input are split along `x` and reduced across it.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let input_sharding =
            sharding(&mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"])]);
        let output_sharding = Sharding::replicated(mesh.clone(), 2);
        let program = trace_program(
            |inputs| {
                vec![
                    shard_map(
                        |x: TestTracer| x.parallel_reduce(ReductionKind::Sum, "x").unwrap(),
                        inputs[0].clone(),
                        mesh.clone(),
                        input_sharding.clone(),
                        output_sharding.clone(),
                    )
                    .unwrap(),
                ]
            },
            vec![f32_type(&[2, 2], None)],
        );
        assert_eq!(
            program.interpret(vec![f32_value(f32_type(&[2, 2], None), &[1.0, 2.0, 3.0, 4.0])]),
            Ok(vec![f32_value(f32_type(&[1, 2], Some(output_sharding)), &[4.0, 6.0])]),
        );
    }

    #[test]
    fn test_shard_map_interpretation_nested_shard_maps() {
        // An outer map over `x` splits a `f32[4]` input into `f32[2]` shards, and an inner map over `y` splits each of
        // those into `f32[1]` shards, so the device at `(x, y)` holds element `2 * x + y`.
        let mesh = manual_mesh_2x2();
        let along_x = sharding(&mesh, vec![ShardingDimension::sharded(["x"])]);
        let along_y = sharding(&mesh, vec![ShardingDimension::sharded(["y"])]);
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let outer_shard_map = |out_sharding: &Sharding| {
            ShardMap::new(mesh.clone(), vec![along_x.clone()], vec![out_sharding.clone()], vec!["x".to_string()])
                .unwrap()
        };
        let outer_local_type = outer_shard_map(&along_x).local_input_type(0, &f32_type(&[4], Some(along_x.clone())));
        let outer_local_type = outer_local_type.unwrap();
        let inner_shard_map =
            ShardMap::new(mesh.clone(), vec![along_y.clone()], vec![along_y.clone()], vec!["y".to_string()]).unwrap();
        let inner_global_type = inner_shard_map.global_input_type(0, &outer_local_type).unwrap();
        let inner_local_type = ArrayIrType::Array(inner_shard_map.local_input_type(0, &inner_global_type).unwrap());
        let nested_program = |inner_body: TestProgram, out_sharding: &Sharding| {
            let outer_local_type = ArrayIrType::Array(outer_local_type.clone());
            let inner_operation =
                ShardMapOperation::from_program(&inner_body, vec![outer_local_type.clone()], inner_shard_map.clone());
            let inner_operation = inner_operation.unwrap();
            let outer_body = build_program(vec![outer_local_type], |builder, inputs| {
                let inner_body = builder.import_program(inner_body);
                builder.add_instruction(inner_operation, vec![inner_body], inputs.to_vec(), None).unwrap().to_vec()
            });
            shard_map_program(outer_body, vec![ArrayIrType::Array(f32_type(&[4], None))], outer_shard_map(out_sharding))
        };
        let input = f32_value(f32_type(&[4], None), &[1.0, 2.0, 3.0, 4.0]);

        // An inner body without collectives over `x` runs once per outer device through the ordinary eager rule.
        let double = build_program(vec![inner_local_type.clone()], |builder, inputs| {
            let add = ArrayOperation::Add(AddOperation::new());
            builder.add_instruction(add, Vec::new(), vec![inputs[0], inputs[0]], None).unwrap().to_vec()
        });
        assert_eq!(
            nested_program(double, &along_x).interpret(vec![input.clone()]),
            Ok(vec![f32_value(f32_type(&[4], Some(along_x.clone())), &[2.0, 4.0, 6.0, 8.0])]),
        );

        // An inner body that reduces over the enclosing axis `x` runs in lockstep on all four devices. The devices
        // `(0, y)` and `(1, y)` hold `1 + y` and `3 + y`, so the reduction yields `[4, 6]` along `y`.
        let reduce = build_program(vec![inner_local_type], |builder, inputs| {
            let operation = ParallelReduceOperation::new(ReductionKind::Sum, "x".to_string()).with_mesh(mesh.clone());
            let operation = ArrayOperation::ParallelReduce(operation);
            builder.add_instruction(operation, Vec::new(), vec![inputs[0]], None).unwrap().to_vec()
        });
        assert_eq!(
            nested_program(reduce, &replicated).interpret(vec![input]),
            Ok(vec![f32_value(f32_type(&[2], Some(replicated.clone())), &[4.0, 6.0])]),
        );
    }

    #[test]
    fn test_shard_map_interpretation_rejects_manual_axes_made_manual_again() {
        // A nested map that makes the manual axis `x` of its enclosing map manual again and that communicates along `x`
        // (so that it must run in lockstep with the enclosing devices) is rejected, because the devices along `x`
        // already run the enclosing body separately. Here, the inputs of both maps are replicated, so type inference,
        // which detects this through the variation of the inputs, accepts the program.
        let mesh = manual_mesh();
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let shard_map =
            ShardMap::new(mesh.clone(), vec![replicated.clone()], vec![replicated.clone()], Vec::new()).unwrap();
        let local_type = ArrayIrType::Array(shard_map.local_input_type(0, &f32_type(&[2], Some(replicated))).unwrap());
        let inner_body = build_program(vec![local_type.clone()], |builder, inputs| {
            let vary = ArrayOperation::ParallelVary(ParallelVaryOperation::new("x".to_string()));
            let value = builder.add_instruction(vary, Vec::new(), vec![inputs[0]], None).unwrap()[0];
            let sum = ParallelReduceOperation::new(ReductionKind::Sum, "x".to_string()).with_mesh(mesh.clone());
            let sum = ArrayOperation::ParallelReduce(sum);
            builder.add_instruction(sum, Vec::new(), vec![value], None).unwrap().to_vec()
        });
        let inner_operation =
            ShardMapOperation::from_program(&inner_body, vec![local_type.clone()], shard_map.clone()).unwrap();
        let outer_body = build_program(vec![local_type], |builder, inputs| {
            let inner_body = builder.import_program(inner_body);
            builder.add_instruction(inner_operation, vec![inner_body], inputs.to_vec(), None).unwrap().to_vec()
        });
        let program = shard_map_program(outer_body, vec![ArrayIrType::Array(f32_type(&[2], None))], shard_map);
        assert_eq!(
            program.interpret(vec![f32_value(f32_type(&[2], None), &[1.0, 2.0])]),
            Err(ProgramError::MalformedProgram(
                "`shard_map` makes manual axis `x` manual again inside an enclosing manual region over it".to_string(),
            )),
        );
    }

    #[test]
    fn test_shard_map_interpretation_while() {
        // A loop whose body reduces over the manual axis runs its iterations in lockstep on both devices, here for its
        // bound of two iterations, each of which replaces the value of every device by the sum over both devices.
        let mesh = manual_mesh();
        let sharded = sharding(&mesh, vec![ShardingDimension::sharded(["x"])]);
        let shard_map = ShardMap::new(mesh.clone(), vec![sharded.clone()], vec![sharded.clone()], Vec::new()).unwrap();
        let local_type = shard_map.local_input_type(0, &f32_type(&[2], Some(sharded.clone()))).unwrap();
        let local_type = ArrayIrType::Array(local_type);
        let condition = build_program(vec![local_type.clone()], |builder, _| {
            vec![builder.add_constant(ArrayIrValue::Array(Array::scalar(true).unwrap()))]
        });
        let reduce =
            build_program(vec![local_type.clone()], |builder, inputs| vec![add_sum_over_x(builder, &mesh, inputs[0])]);
        let body = build_program(vec![local_type], |builder, inputs| {
            let regions = vec![builder.import_program(condition), builder.import_program(reduce)];
            let operation = WhileOperation::<ArrayIrType>::new().with_iteration_bound(2).unwrap();
            builder.add_instruction(operation, regions, inputs.to_vec(), None).unwrap().to_vec()
        });
        let program = shard_map_program(body, vec![ArrayIrType::Array(f32_type(&[2], None))], shard_map);
        assert_eq!(
            program.interpret(vec![f32_value(f32_type(&[2], None), &[1.0, 2.0])]),
            Ok(vec![f32_value(f32_type(&[2], Some(sharded)), &[6.0, 6.0])]),
        );
    }

    #[test]
    fn test_shard_map_interpretation_while_rejects_divergent_predicates() {
        // A loop whose body reduces over the manual axis and whose predicate (here, whether the local value is
        // positive) holds on device 0 but not on device 1 would reach the reduction with mismatched participants.
        let mesh = manual_mesh();
        let sharded = sharding(&mesh, vec![ShardingDimension::sharded(["x"])]);
        let shard_map = ShardMap::new(mesh.clone(), vec![sharded.clone()], vec![sharded.clone()], Vec::new()).unwrap();
        let local_type = shard_map.local_input_type(0, &f32_type(&[2], Some(sharded))).unwrap();
        let local_type = ArrayIrType::Array(local_type);
        let condition = build_program(vec![local_type.clone()], |builder, inputs| {
            let reshape = ArrayOperation::Reshape(ReshapeOperation::new(Shape::new(Vec::new())));
            let value = builder.add_instruction(reshape, Vec::new(), vec![inputs[0]], None).unwrap()[0];
            let zero = ArrayOperation::ZeroLike(ZeroLikeOperation::new());
            let zero = builder.add_instruction(zero, Vec::new(), vec![value], None).unwrap()[0];
            let compare = ArrayOperation::Compare(CompareOperation::new(ComparisonDirection::GreaterThan));
            builder.add_instruction(compare, Vec::new(), vec![value, zero], None).unwrap().to_vec()
        });
        let reduce =
            build_program(vec![local_type.clone()], |builder, inputs| vec![add_sum_over_x(builder, &mesh, inputs[0])]);
        let body = build_program(vec![local_type], |builder, inputs| {
            let regions = vec![builder.import_program(condition), builder.import_program(reduce)];
            builder
                .add_instruction(WhileOperation::<ArrayIrType>::new(), regions, inputs.to_vec(), None)
                .unwrap()
                .to_vec()
        });
        let program = shard_map_program(body, vec![ArrayIrType::Array(f32_type(&[2], None))], shard_map);
        assert_eq!(
            program.interpret(vec![f32_value(f32_type(&[2], None), &[1.0, -1.0])]),
            Err(ProgramError::UnsupportedOperation {
                message: "the devices of a `shard_map` disagree on the predicate of a `while` whose regions use \
                          collectives over the manual axes, so the collectives would not have matching participants"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_shard_map_interpretation_scan() {
        // A scan whose body reduces over the manual axis runs its iterations in lockstep on both devices. Every
        // iteration replaces the carry of each device by the sum of the carries of both devices and stacks it.
        let mesh = manual_mesh();
        let sharded = sharding(&mesh, vec![ShardingDimension::sharded(["x"])]);
        let stacked = sharding(&mesh, vec![ShardingDimension::Replicated, ShardingDimension::sharded(["x"])]);
        let shard_map =
            ShardMap::new(mesh.clone(), vec![sharded.clone()], vec![sharded.clone(), stacked.clone()], Vec::new());
        let shard_map = shard_map.unwrap();
        let local_type = shard_map.local_input_type(0, &f32_type(&[2], Some(sharded.clone()))).unwrap();
        let index_type = ArrayIrType::Array(ArrayType::scalar(DataType::I64));
        let scan_body = build_program(vec![index_type, ArrayIrType::Array(local_type.clone())], |builder, inputs| {
            let carry = add_sum_over_x(builder, &mesh, inputs[1]);
            vec![carry, carry]
        });
        let body = build_program(vec![ArrayIrType::Array(local_type)], |builder, inputs| {
            let scan_body = builder.import_program(scan_body);
            let scan = ScanOperation::<ArrayIrType>::new(1, 2usize);
            builder.add_instruction(scan, vec![scan_body], vec![inputs[0]], None).unwrap().to_vec()
        });
        let program = shard_map_program(body, vec![ArrayIrType::Array(f32_type(&[2], None))], shard_map);
        assert_eq!(
            program.interpret(vec![f32_value(f32_type(&[2], None), &[1.0, 2.0])]),
            Ok(vec![
                f32_value(f32_type(&[2], Some(sharded)), &[6.0, 6.0]),
                f32_value(f32_type(&[2, 2], Some(stacked)), &[3.0, 3.0, 6.0, 6.0]),
            ]),
        );
    }

    #[test]
    fn test_shard_map_interpretation_scan_reference_carries() {
        // A scan whose reference carry each device overwrites with the sum over both devices runs in lockstep, every
        // device owns its shard of the caller's reference, and the final shards are written back into that reference.
        let mesh = manual_mesh();
        let sharded = sharding(&mesh, vec![ShardingDimension::sharded(["x"])]);
        let shard_map = ShardMap::new(mesh.clone(), vec![sharded.clone()], vec![sharded.clone()], Vec::new()).unwrap();
        let local_type = shard_map.local_input_type(0, &f32_type(&[2], Some(sharded))).unwrap();
        let reference_type = ArrayIrType::Reference(ReferenceType::new(local_type));
        let index_type = ArrayIrType::Array(ArrayType::scalar(DataType::I64));
        let scan_body = build_program(vec![index_type, reference_type.clone()], |builder, inputs| {
            let value = builder.add_instruction(reference_read(), Vec::new(), vec![inputs[1]], None).unwrap()[0];
            let sum = add_sum_over_x(builder, &mesh, value);
            let write = ReferenceWriteOperation::<ArrayType, ArrayIrType, ArrayReferenceTransform>::new();
            builder.add_instruction(write, Vec::new(), vec![inputs[1], sum], None).unwrap();
            vec![inputs[1]]
        });
        let body = build_program(vec![reference_type], |builder, inputs| {
            let scan_body = builder.import_program(scan_body);
            let scan = ScanOperation::<ArrayIrType>::new(1, 2usize);
            builder.add_instruction(scan, vec![scan_body], vec![inputs[0]], None).unwrap().to_vec()
        });
        let global_type = f32_type(&[2], None);
        let program =
            shard_map_program(body, vec![ArrayIrType::Reference(ReferenceType::new(global_type.clone()))], shard_map);
        let reference = ArrayReference::new(Array::from_elements(global_type.clone(), &[1.0f32, 2.0]).unwrap());
        assert_eq!(
            program.interpret(vec![ArrayIrValue::Reference(reference.clone())]),
            Ok(vec![ArrayIrValue::Reference(reference.clone())]),
        );
        assert_eq!(reference.read(), Ok(Array::from_elements(global_type, &[6.0f32, 6.0]).unwrap()));
    }

    #[test]
    fn test_shard_map_interpretation_scan_rejects_dynamic_lengths() {
        // A scan whose body reduces over the manual axis runs in lockstep only over a static trip count. Here, the trip
        // count is the first-class dimension `length` that the body derives from its replicated input `count`, which
        // stacks the local value of each device `count` times.
        let mesh = manual_mesh();
        let sharded = sharding(&mesh, vec![ShardingDimension::sharded(["x"])]);
        let replicated = Sharding::replicated(mesh.clone(), 0);
        let shard_map =
            ShardMap::new(mesh.clone(), vec![sharded.clone(), replicated.clone()], vec![sharded.clone()], Vec::new());
        let shard_map = shard_map.unwrap();
        let local_type = ArrayIrType::Array(shard_map.local_input_type(0, &f32_type(&[2], Some(sharded))).unwrap());
        let count_type = ArrayIrType::Array(shard_map.local_input_type(1, &f32_type(&[], Some(replicated))).unwrap());
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(4)).unwrap());
        let varying = Sharding::replicated(mesh.clone(), 0).with_varying_manual_axes(["x"]).unwrap();
        let value_type = ArrayIrType::Array(f32_type(&[], Some(varying)));
        let body = build_program(vec![local_type, count_type], |builder, inputs| {
            let reshape = ArrayOperation::Reshape(ReshapeOperation::new(Shape::new(Vec::new())));
            let value = builder.add_instruction(reshape, Vec::new(), vec![inputs[0]], None).unwrap()[0];
            let convert = ArrayOperation::ConvertElementType(ConvertElementTypeOperation::new(DataType::I64, false));
            let count = builder.add_instruction(convert, Vec::new(), vec![inputs[1]], None).unwrap()[0];
            let dimension = DimensionFromScalarOperation::new(length.clone());
            let dimension = builder.add_instruction(dimension, Vec::new(), vec![count], None).unwrap()[0];
            let broadcast = DynamicBroadcastOperation::new(Vec::new());
            let stack = builder.add_instruction(broadcast, Vec::new(), vec![value, dimension], None).unwrap()[0];
            let scan_body = build_program(
                vec![ArrayIrType::Array(ArrayType::scalar(DataType::I64)), value_type.clone(), value_type.clone()],
                |builder, inputs| vec![add_sum_over_x(builder, &mesh, inputs[1])],
            );
            let scan_body = builder.import_program(scan_body);
            let scan = ScanOperation::<ArrayIrType>::new(1, Dimension::Dynamic(length.clone()));
            let carry = builder.add_instruction(scan, vec![scan_body], vec![value, stack, dimension], None).unwrap()[0];
            let reshape = ArrayOperation::Reshape(ReshapeOperation::new(Shape::new(vec![Dimension::Static(1)])));
            builder.add_instruction(reshape, Vec::new(), vec![carry], None).unwrap().to_vec()
        });
        let input_types = vec![ArrayIrType::Array(f32_type(&[2], None)), ArrayIrType::Array(f32_type(&[], None))];
        let program = shard_map_program(body, input_types, shard_map);
        assert_eq!(
            program
                .interpret(vec![f32_value(f32_type(&[2], None), &[1.0, 2.0]), f32_value(f32_type(&[], None), &[2.0])]),
            Err(ProgramError::UnsupportedOperation {
                message: "`shard_map` interpretation cannot run `scan` with a dynamic length in lockstep across \
                          devices"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_shard_map_interpretation_condition() {
        // A condition whose branches reduce over the manual axis runs the branch that both devices select in lockstep.
        let mesh = manual_mesh();
        let sharded = sharding(&mesh, vec![ShardingDimension::sharded(["x"])]);
        let shard_map =
            ShardMap::new(mesh.clone(), vec![sharded.clone(), sharded.clone()], vec![sharded.clone()], Vec::new());
        let shard_map = shard_map.unwrap();
        let predicate_type = shard_map
            .local_input_type(0, &ArrayType::new_static(DataType::Boolean, [2]).with_sharding(sharded.clone()).unwrap())
            .unwrap();
        let value_type =
            ArrayIrType::Array(shard_map.local_input_type(1, &f32_type(&[2], Some(sharded.clone()))).unwrap());
        let reduce_branch =
            build_program(vec![value_type.clone()], |builder, inputs| vec![add_sum_over_x(builder, &mesh, inputs[0])]);
        let identity_branch = build_program(vec![value_type.clone()], |_, inputs| inputs.to_vec());
        let body = build_program(vec![ArrayIrType::Array(predicate_type), value_type], |builder, inputs| {
            let reshape = ArrayOperation::Reshape(ReshapeOperation::new(Shape::new(Vec::new())));
            let predicate = builder.add_instruction(reshape, Vec::new(), vec![inputs[0]], None).unwrap()[0];
            let branches = vec![builder.import_program(reduce_branch), builder.import_program(identity_branch)];
            let condition = ConditionOperation::<ArrayIrType>::new();
            builder.add_instruction(condition, branches, vec![predicate, inputs[1]], None).unwrap().to_vec()
        });
        let input_types = vec![
            ArrayIrType::Array(ArrayType::new_static(DataType::Boolean, [2])),
            ArrayIrType::Array(f32_type(&[2], None)),
        ];
        let program = shard_map_program(body, input_types, shard_map);
        let predicates = |predicates: &[bool]| {
            ArrayIrValue::Array(
                Array::from_elements(ArrayType::new_static(DataType::Boolean, [2]), predicates).unwrap(),
            )
        };
        let values = f32_value(f32_type(&[2], None), &[1.0, 2.0]);
        assert_eq!(
            program.interpret(vec![predicates(&[true, true]), values.clone()]),
            Ok(vec![f32_value(f32_type(&[2], Some(sharded.clone())), &[3.0, 3.0])]),
        );
        assert_eq!(
            program.interpret(vec![predicates(&[false, false]), values.clone()]),
            Ok(vec![f32_value(f32_type(&[2], Some(sharded)), &[1.0, 2.0])]),
        );

        // Devices that select different branches would reach the reduction with mismatched participants.
        assert_eq!(
            program.interpret(vec![predicates(&[true, false]), values]),
            Err(ProgramError::UnsupportedOperation {
                message: "the devices of a `shard_map` disagree on the predicate of a `condition` whose regions use \
                          collectives over the manual axes, so the collectives would not have matching participants"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_shard_map_interpretation_replays_single_region_wrappers() {
        // A `custom_function`, a `linear_call`, and a `rematerialize` call whose primal (i.e., first) regions reduce
        // over the manual axis replay those regions in lockstep on both devices, as their eager rules replay them on
        // one device. Each one replaces the values `1` and `2` of the two devices by their sum, which doubles every
        // time.
        let mesh = manual_mesh();
        let sharded = sharding(&mesh, vec![ShardingDimension::sharded(["x"])]);
        let shard_map = ShardMap::new(mesh.clone(), vec![sharded.clone()], vec![sharded.clone()], Vec::new()).unwrap();
        let local_type =
            ArrayIrType::Array(shard_map.local_input_type(0, &f32_type(&[2], Some(sharded.clone()))).unwrap());
        let reduce = || {
            build_program(vec![local_type.clone()], |builder, inputs| vec![add_sum_over_x(builder, &mesh, inputs[0])])
        };
        let identity = build_program(vec![local_type.clone()], |_, inputs| inputs.to_vec());
        let body = build_program(vec![local_type.clone()], |builder, inputs| {
            let primal = builder.import_program(reduce());
            let custom_function = CustomFunctionOperation::<TestValue, TestOperation>::from_rule_regions(
                CustomFunctionJvpRule::Primal,
                false,
            );
            let value = builder.add_instruction(custom_function, vec![primal], inputs.to_vec(), None).unwrap()[0];
            let regions = vec![builder.import_program(reduce()), builder.import_program(identity)];
            let linear_call = LinearCallOperation::<ArrayIrType>::new(0);
            let value = builder.add_instruction(linear_call, regions, vec![value], None).unwrap()[0];
            let body = builder.import_program(reduce());
            let rematerialize =
                RematerializeOperation::<ArrayIrType>::new(ResidualPolicyReference::new(NothingSavable));
            builder.add_instruction(rematerialize, vec![body], vec![value], None).unwrap().to_vec()
        });
        let program = shard_map_program(body, vec![ArrayIrType::Array(f32_type(&[2], None))], shard_map);
        assert_eq!(
            program.interpret(vec![f32_value(f32_type(&[2], None), &[1.0, 2.0])]),
            Ok(vec![f32_value(f32_type(&[2], Some(sharded)), &[12.0, 12.0])]),
        );
    }

    #[test]
    fn test_shard_map_interpretation_rejects_other_region_operations() {
        // A region operation whose regions reduce over the manual axis needs lockstep execution, which the emulator
        // provides only for the region operations whose region sequencing it knows. Here, the backward rule region of a
        // `custom_function_transpose` carrier reduces over the manual axis.
        let mesh = manual_mesh();
        let sharded = sharding(&mesh, vec![ShardingDimension::sharded(["x"])]);
        let shard_map = ShardMap::new(mesh.clone(), vec![sharded.clone()], vec![sharded], Vec::new()).unwrap();
        let local_type = ArrayIrType::Array(shard_map.local_input_type(0, &f32_type(&[2], None)).unwrap());
        let backward =
            build_program(vec![local_type.clone()], |builder, inputs| vec![add_sum_over_x(builder, &mesh, inputs[0])]);
        let body = build_program(vec![local_type.clone()], |builder, inputs| {
            let backward = builder.import_program(backward);
            let carrier = CustomFunctionTransposeOperation::<TestValue, TestOperation>::from_backward_region(
                0,
                vec![local_type.clone()],
                vec![local_type.clone()],
            );
            builder.add_instruction(carrier, vec![backward], inputs.to_vec(), None).unwrap().to_vec()
        });
        let program = shard_map_program(body, vec![ArrayIrType::Array(f32_type(&[2], None))], shard_map);
        assert_eq!(
            program.interpret(vec![f32_value(f32_type(&[2], None), &[1.0, 2.0])]),
            Err(ProgramError::UnsupportedOperation {
                message: "`shard_map` interpretation cannot run `custom_function_transpose` in lockstep across \
                          devices, which its attached regions require because they use collectives over the manual \
                          axes"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_shard_map_interpretation_parallel_all_gather() {
        // Every device gathers the tiles of both devices, so the tiled output holds one gathered copy per device.
        let mesh = manual_mesh();
        let sharded = sharding(&mesh, vec![ShardingDimension::sharded(["x"])]);
        let gather_rows = sharding(&mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::Replicated]);
        let program = trace_program(
            |inputs| {
                vec![
                    shard_map(
                        |x: TestTracer| x.parallel_all_gather_tiled("x", 0usize).unwrap(),
                        inputs[0].clone(),
                        mesh.clone(),
                        sharded.clone(),
                        sharded.clone(),
                    )
                    .unwrap(),
                    shard_map(
                        |x: TestTracer| x.parallel_all_gather("x", 0usize).unwrap(),
                        inputs[0].clone(),
                        mesh.clone(),
                        sharded.clone(),
                        gather_rows.clone(),
                    )
                    .unwrap(),
                ]
            },
            vec![f32_type(&[4], None)],
        );
        assert_eq!(
            program.interpret(vec![f32_value(f32_type(&[4], None), &[1.0, 2.0, 3.0, 4.0])]),
            Ok(vec![
                f32_value(f32_type(&[8], Some(sharded)), &[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]),
                f32_value(f32_type(&[4, 2], Some(gather_rows)), &[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]),
            ]),
        );
    }

    #[test]
    fn test_shard_map_interpretation_parallel_sum_scatter() {
        // Every device receives its tile of the sum of the inputs of both devices: the sum `[6, 8, 10, 12]` of the
        // tiles `[1, 2, 3, 4]` and `[5, 6, 7, 8]` is scattered, so device 0 receives `[6, 8]` and device 1 `[10, 12]`.
        let mesh = manual_mesh();
        let sharded = sharding(&mesh, vec![ShardingDimension::sharded(["x"])]);
        let program = trace_program(
            |inputs| {
                vec![
                    shard_map(
                        |x: TestTracer| x.parallel_sum_scatter_tiled("x", 0usize).unwrap(),
                        inputs[0].clone(),
                        mesh.clone(),
                        sharded.clone(),
                        sharded.clone(),
                    )
                    .unwrap(),
                ]
            },
            vec![f32_type(&[8], None)],
        );
        assert_eq!(
            program.interpret(vec![f32_value(f32_type(&[8], None), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])]),
            Ok(vec![f32_value(f32_type(&[4], Some(sharded)), &[6.0, 8.0, 10.0, 12.0])]),
        );
    }

    #[test]
    fn test_shard_map_interpretation_validates_collective_result_extents() {
        // The mixed form of a shape-changing collective receives its result extents as trailing first-class dimension
        // inputs, which the emulator checks against the results that it computes. Here, the body derives the extent
        // `length` of a scattered sum of `f32[4]` tiles from its replicated input, which must therefore be `2`.
        let mesh = manual_mesh();
        let sharded = sharding(&mesh, vec![ShardingDimension::sharded(["x"])]);
        let replicated = Sharding::replicated(mesh.clone(), 0);
        let shard_map =
            ShardMap::new(mesh.clone(), vec![sharded.clone(), replicated.clone()], vec![sharded.clone()], Vec::new());
        let shard_map = shard_map.unwrap();
        let local_type = shard_map.local_input_type(0, &f32_type(&[8], Some(sharded.clone()))).unwrap();
        let extent_type = shard_map.local_input_type(1, &f32_type(&[], Some(replicated))).unwrap();
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(4)).unwrap());
        let body =
            build_program(vec![ArrayIrType::Array(local_type), ArrayIrType::Array(extent_type)], |builder, inputs| {
                let convert =
                    ArrayOperation::ConvertElementType(ConvertElementTypeOperation::new(DataType::I64, false));
                let extent = builder.add_instruction(convert, Vec::new(), vec![inputs[1]], None).unwrap()[0];
                let dimension = DimensionFromScalarOperation::new(length);
                let dimension = builder.add_instruction(dimension, Vec::new(), vec![extent], None).unwrap()[0];
                let scatter = ParallelSumScatterOperation::new("x".to_string(), 2, 0, CollectiveOptions::tiled())
                    .with_mesh(mesh.clone());
                let sum = builder.add_instruction(scatter, Vec::new(), vec![inputs[0], dimension], None).unwrap()[0];
                let reduce = ArrayOperation::Reduce(ReduceOperation::new(vec![0], ReductionKind::Sum));
                let sum = builder.add_instruction(reduce, Vec::new(), vec![sum], None).unwrap()[0];
                let reshape = ArrayOperation::Reshape(ReshapeOperation::new(Shape::new(vec![Dimension::Static(1)])));
                builder.add_instruction(reshape, Vec::new(), vec![sum], None).unwrap().to_vec()
            });
        let input_types = vec![ArrayIrType::Array(f32_type(&[8], None)), ArrayIrType::Array(f32_type(&[], None))];
        let program = shard_map_program(body, input_types, shard_map);
        let input = f32_value(f32_type(&[8], None), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(
            program.interpret(vec![input.clone(), f32_value(f32_type(&[], None), &[2.0])]),
            Ok(vec![f32_value(f32_type(&[2], Some(sharded)), &[14.0, 22.0])]),
        );
        assert_eq!(
            program.interpret(vec![input, f32_value(f32_type(&[], None), &[3.0])]),
            Err(ProgramError::InvalidArgument {
                message: "`parallel_sum_scatter` result extent input #0 is `3`, but result axis #0 has \
                          extent 2"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_shard_map_interpretation_parallel_all_to_all() {
        // The device at position `p` receives chunk `p` of the columns of both devices and concatenates them along
        // the rows, which moves a row-sharded matrix to a column-sharded one.
        let mesh = manual_mesh();
        let rows = sharding(&mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::Replicated]);
        let columns = sharding(&mesh, vec![ShardingDimension::Replicated, ShardingDimension::sharded(["x"])]);
        let program = trace_program(
            |inputs| {
                vec![
                    shard_map(
                        |x: TestTracer| x.parallel_all_to_all_tiled("x", 1usize, 0usize).unwrap(),
                        inputs[0].clone(),
                        mesh.clone(),
                        rows.clone(),
                        columns.clone(),
                    )
                    .unwrap(),
                ]
            },
            vec![f32_type(&[2, 2], None)],
        );
        assert_eq!(
            program.interpret(vec![f32_value(f32_type(&[2, 2], None), &[1.0, 2.0, 3.0, 4.0])]),
            Ok(vec![f32_value(f32_type(&[2, 2], Some(columns)), &[1.0, 2.0, 3.0, 4.0])]),
        );
    }

    #[test]
    fn test_shard_map_interpretation_parallel_permute() {
        // Device 1 receives the tile of device 0, while device 0, which no pair targets, receives zeros.
        let mesh = manual_mesh();
        let sharded = sharding(&mesh, vec![ShardingDimension::sharded(["x"])]);
        let program = trace_program(
            |inputs| {
                vec![
                    shard_map(
                        |x: TestTracer| x.parallel_permute("x", vec![(0, 1)]).unwrap(),
                        inputs[0].clone(),
                        mesh.clone(),
                        sharded.clone(),
                        sharded.clone(),
                    )
                    .unwrap(),
                ]
            },
            vec![f32_type(&[2], None)],
        );
        assert_eq!(
            program.interpret(vec![f32_value(f32_type(&[2], None), &[1.0, 2.0])]),
            Ok(vec![f32_value(f32_type(&[2], Some(sharded)), &[0.0, 1.0])]),
        );
    }

    #[test]
    fn test_shard_map_interpretation_parallel_ragged_all_to_all() {
        // Device 0 sends `[1]` to itself and `[2, 3]` to device 1, and device 1 sends `[4, 5]` to device 0 and `[6]` to
        // itself, each at the receiver's output offsets, so device 0 receives `[1, 4, 5]` and device 1 receives
        // `[2, 3, 6]`, while the last row of both output seeds passes through unchanged.
        let mesh = manual_mesh();
        let sharded = sharding(&mesh, vec![ShardingDimension::sharded(["x"])]);
        let metadata_type = ArrayType::new_static(DataType::I32, [4]);
        let program = trace_program(
            |inputs| {
                vec![
                    shard_map(
                        |inputs: Vec<TestTracer>| {
                            inputs[0]
                                .parallel_ragged_all_to_all(
                                    "x", &inputs[1], &inputs[2], &inputs[3], &inputs[4], &inputs[5],
                                )
                                .unwrap()
                        },
                        inputs,
                        mesh.clone(),
                        vec![sharded.clone(); 6],
                        sharded.clone(),
                    )
                    .unwrap(),
                ]
            },
            vec![
                f32_type(&[6], None),
                f32_type(&[8], None),
                metadata_type.clone(),
                metadata_type.clone(),
                metadata_type.clone(),
                metadata_type.clone(),
            ],
        );
        let metadata =
            |elements: &[i32]| ArrayIrValue::Array(Array::from_elements(metadata_type.clone(), elements).unwrap());
        assert_eq!(
            program.interpret(vec![
                f32_value(f32_type(&[6], None), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                f32_value(f32_type(&[8], None), &[-1.0; 8]),
                metadata(&[0, 1, 0, 2]),
                metadata(&[1, 2, 2, 1]),
                metadata(&[0, 0, 1, 2]),
                metadata(&[1, 2, 2, 1]),
            ]),
            Ok(vec![f32_value(f32_type(&[8], Some(sharded)), &[1.0, 4.0, 5.0, -1.0, 2.0, 3.0, 6.0, -1.0])]),
        );
    }

    #[test]
    fn test_shard_map_emulator_run_collective_parallel_ragged_all_to_all() {
        // Participant `p` holds the rows `[10·p + 1, 10·p + 2]` and sends row `i` to the member at position `i` of its
        // group, which places it at the row given by the sender's own position. With groups `[[0, 2], [3, 1]]`, the
        // members exchange only within their group, and the last row of every output seed passes through unchanged.
        let context = EagerContext::<TestValue, TestOperation>::new();
        let driver = EmptyRegionDriver;
        let emulator = ShardMapEmulator {
            context: &context,
            driver: &driver,
            grid: ManualDeviceGrid { axes: vec![("x".to_string(), 4)] },
        };
        let operation = TestOperation::ParallelRaggedAllToAll(
            ParallelRaggedAllToAllOperation::grouped("x".to_string(), 4, vec![vec![0, 2], vec![3, 1]]).unwrap(),
        );
        let inputs = vec![
            vec![
                f32_value(f32_type(&[2], None), &[1.0, 2.0]),
                f32_value(f32_type(&[2], None), &[11.0, 12.0]),
                f32_value(f32_type(&[2], None), &[21.0, 22.0]),
                f32_value(f32_type(&[2], None), &[31.0, 32.0]),
            ],
            vec![f32_value(f32_type(&[3], None), &[-1.0; 3]); 4],
            vec![ArrayIrValue::Array(Array::vector(vec![0i64, 1]).unwrap()); 4],
            vec![ArrayIrValue::Array(Array::vector(vec![1i64, 1]).unwrap()); 4],
            vec![
                ArrayIrValue::Array(Array::vector(vec![0i64, 0]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![1i64, 1]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![1i64, 1]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![0i64, 0]).unwrap()),
            ],
            vec![ArrayIrValue::Array(Array::vector(vec![1i64, 1]).unwrap()); 4],
        ];
        assert_eq!(
            emulator.run_collective(&operation, &inputs),
            Ok(Some(vec![vec![
                f32_value(f32_type(&[3], None), &[1.0, 21.0, -1.0]),
                f32_value(f32_type(&[3], None), &[32.0, 12.0, -1.0]),
                f32_value(f32_type(&[3], None), &[2.0, 22.0, -1.0]),
                f32_value(f32_type(&[3], None), &[31.0, 11.0, -1.0]),
            ]])),
        );

        // The staged participant count must agree with the emulated axis, whose inputs cover every device.
        let smaller_emulator = ShardMapEmulator {
            context: &context,
            driver: &driver,
            grid: ManualDeviceGrid { axes: vec![("x".to_string(), 2)] },
        };
        let fewer_inputs = inputs.iter().map(|values| values[..2].to_vec()).collect::<Vec<_>>();
        assert_eq!(
            smaller_emulator.run_collective(&operation, &fewer_inputs),
            Err(ProgramError::MalformedProgram(
                "`parallel_ragged_all_to_all` was staged for 4 participants, but manual axis `x` has 2 devices"
                    .to_string(),
            )),
        );

        // The emulator requires all six exchange inputs, each containing one value per device.
        assert_eq!(
            emulator.run_collective(&operation, &inputs[..5]),
            Err(ProgramError::InvalidInputCount { expected: 6, actual: 5 }),
        );

        // Corresponding participant inputs must share one static shape and data type, even when each participant's
        // own inputs describe a valid exchange.
        let mut longer_inputs = inputs.clone();
        longer_inputs[0][1] = f32_value(f32_type(&[3], None), &[11.0, 12.0, 13.0]);
        assert_eq!(
            emulator.run_collective(&operation, &longer_inputs),
            Err(TypeError::invalid(
                "`parallel_ragged_all_to_all` participant inputs 0 must share one static shape and data type but got \
                 `f32[2]` and `f32[3]`",
            )
            .into()),
        );
        let mut wider_inputs = inputs.clone();
        wider_inputs[0][1] = ArrayIrValue::Array(Array::vector(vec![11.0f64, 12.0]).unwrap());
        wider_inputs[1][1] = ArrayIrValue::Array(Array::vector(vec![-1.0f64; 3]).unwrap());
        assert_eq!(
            emulator.run_collective(&operation, &wider_inputs),
            Err(TypeError::invalid(
                "`parallel_ragged_all_to_all` participant inputs 0 must share one static shape and data type but got \
                 `f32[2]` and `f64[2]`",
            )
            .into()),
        );

        // Packing the physical exchange requires static shapes even when the logical exchange types are valid.
        let extent = DimensionVariable::new("extent", DimensionBounds::new(0, Some(4)).unwrap());
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent)]));
        let mut dynamic_inputs = inputs;
        for input in &mut dynamic_inputs[0] {
            let ArrayIrValue::Array(array) = input else { unreachable!() };
            let bytes = array.logical_bytes();
            *input = ArrayIrValue::Array(Array::new_unchecked(dynamic_type.clone(), Arc::new(bytes)));
        }
        assert_eq!(
            emulator.run_collective(&operation, &dynamic_inputs),
            Err(TypeError::invalid(
                "`parallel_ragged_all_to_all` participant input 0 must have a static shape but got `f32[extent]`",
            )
            .into()),
        );
    }

    #[test]
    fn test_shard_map_interpretation_unreduced() {
        // An input that is unreduced along the manual axis stores its pending sum, which device 0 receives while
        // device 1 receives zeros, and an unreduced output sums the partial outputs of both devices.
        let mesh = manual_mesh();
        let unreduced = Sharding::replicated(mesh.clone(), 1).with_unreduced_axes(["x"]).unwrap();
        let unreduced_type = f32_type(&[2], Some(unreduced.clone()));
        let program = trace_program(
            |inputs| {
                vec![
                    shard_map(
                        |x: TestTracer| x.clone() + x,
                        inputs[0].clone(),
                        mesh.clone(),
                        unreduced.clone(),
                        unreduced.clone(),
                    )
                    .unwrap(),
                ]
            },
            vec![unreduced_type.clone()],
        );
        assert_eq!(
            program.interpret(vec![f32_value(unreduced_type.clone(), &[1.0, 2.0])]),
            Ok(vec![f32_value(unreduced_type.clone(), &[2.0, 4.0])]),
        );

        // The partial outputs of distinct devices are summed elementwise.
        let shard_map = ShardMap::new(mesh, vec![unreduced.clone()], vec![unreduced], Vec::new()).unwrap();
        let local_type = shard_map.local_input_type(0, &unreduced_type).unwrap();
        let grid = ManualDeviceGrid { axes: vec![("x".to_string(), 2)] };
        let partials = [f32_value(local_type.clone(), &[1.0, 2.0]), f32_value(local_type, &[10.0, 20.0])];
        assert_eq!(
            assemble(&shard_map.out_shardings()[0], &unreduced_type, &grid, &partials).map(ArrayIrValue::Array),
            Ok(f32_value(unreduced_type, &[11.0, 22.0])),
        );
    }

    #[test]
    fn test_shard_map_interpretation_references() {
        // Each device adds its tile of the update into its own shard of the reference and reads it back, the caller's
        // reference receives the updated shards of both devices, and the forwarded reference output is the caller's
        // reference itself.
        let mesh = manual_mesh();
        let sharded = sharding(&mesh, vec![ShardingDimension::sharded(["x"])]);
        let shard_map = ShardMap::new(mesh, vec![sharded.clone(); 2], vec![sharded.clone(); 2], Vec::new()).unwrap();
        let local_type = shard_map.local_input_type(0, &f32_type(&[4], Some(sharded.clone()))).unwrap();
        let body = build_program(
            vec![ArrayIrType::Reference(ReferenceType::new(local_type.clone())), ArrayIrType::Array(local_type)],
            |builder, inputs| {
                let add_update = ReferenceAddUpdateOperation::<ArrayType, ArrayIrType, ArrayReferenceTransform>::new();
                builder.add_instruction(add_update, Vec::new(), inputs.to_vec(), None).unwrap();
                let value = builder.add_instruction(reference_read(), Vec::new(), vec![inputs[0]], None).unwrap()[0];
                vec![value, inputs[0]]
            },
        );
        let global_type = f32_type(&[4], None);
        let input_types = vec![
            ArrayIrType::Reference(ReferenceType::new(global_type.clone())),
            ArrayIrType::Array(global_type.clone()),
        ];
        let program = shard_map_program(body, input_types, shard_map);
        let reference = Array::from_elements(global_type.clone(), &[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let reference = ArrayReference::new(reference);
        assert_eq!(
            program.interpret(vec![
                ArrayIrValue::Reference(reference.clone()),
                f32_value(global_type.clone(), &[10.0, 20.0, 30.0, 40.0]),
            ]),
            Ok(vec![
                f32_value(f32_type(&[4], Some(sharded)), &[11.0, 22.0, 33.0, 44.0]),
                ArrayIrValue::Reference(reference.clone()),
            ]),
        );
        assert_eq!(reference.read(), Ok(Array::from_elements(global_type, &[11.0f32, 22.0, 33.0, 44.0]).unwrap()));
    }

    #[test]
    fn test_shard_map_interpretation_rejects_repeated_reference_inputs() {
        // Every reference input is copied into fresh per-device roots and written back independently, so passing one
        // allocation as both reference inputs of a body that writes through the first one and reads through the second
        // one would lose that write. It is rejected instead, as reference discharge rejects it, and the caller's
        // reference is unchanged.
        let mesh = manual_mesh();
        let sharded = sharding(&mesh, vec![ShardingDimension::sharded(["x"])]);
        let shard_map = ShardMap::new(mesh, vec![sharded.clone(); 2], vec![sharded.clone()], Vec::new()).unwrap();
        let local_type = shard_map.local_input_type(0, &f32_type(&[4], Some(sharded))).unwrap();
        let reference_type = ArrayIrType::Reference(ReferenceType::new(local_type));
        let body = build_program(vec![reference_type.clone(), reference_type], |builder, inputs| {
            let value = builder.add_instruction(reference_read(), Vec::new(), vec![inputs[0]], None).unwrap()[0];
            let add_update = ReferenceAddUpdateOperation::<ArrayType, ArrayIrType, ArrayReferenceTransform>::new();
            builder.add_instruction(add_update, Vec::new(), vec![inputs[0], value], None).unwrap();
            builder.add_instruction(reference_read(), Vec::new(), vec![inputs[1]], None).unwrap().to_vec()
        });
        let global_type = f32_type(&[4], None);
        let global_reference_type = ArrayIrType::Reference(ReferenceType::new(global_type.clone()));
        let operation =
            ShardMapOperation::from_program(&body, vec![global_reference_type.clone(); 2], shard_map).unwrap();
        let program = build_program(vec![global_reference_type], |builder, inputs| {
            let body = builder.import_program(body);
            builder.add_instruction(operation, vec![body], vec![inputs[0], inputs[0]], None).unwrap().to_vec()
        });
        let state = Array::from_elements(global_type, &[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let reference = ArrayReference::new(state.clone());
        let expected = ShardMapError::RepeatedReferenceInputAllocation { first_input_index: 0, second_input_index: 1 };
        assert!(matches!(
            program.interpret(vec![ArrayIrValue::Reference(reference.clone())]),
            Err(error) if error.downcast_custom::<ShardMapError>() == Some(&expected),
        ));
        assert_eq!(reference.read(), Ok(state));
    }

    #[test]
    fn test_shard_map_interpretation_replicated_references() {
        // Every device reads its own copy of a reference that is replicated along the manual axis and adds it to its
        // tile of the input, while the caller's reference, which the body only reads, is unchanged.
        let mesh = manual_mesh();
        let sharded = sharding(&mesh, vec![ShardingDimension::sharded(["x"])]);
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let shard_map =
            ShardMap::new(mesh, vec![replicated.clone(), sharded.clone()], vec![sharded.clone()], Vec::new()).unwrap();
        let reference_type = shard_map.local_input_type(0, &f32_type(&[2], Some(replicated))).unwrap();
        let value_type = shard_map.local_input_type(1, &f32_type(&[4], Some(sharded.clone()))).unwrap();
        let body = build_program(
            vec![ArrayIrType::Reference(ReferenceType::new(reference_type)), ArrayIrType::Array(value_type)],
            |builder, inputs| {
                let value = builder.add_instruction(reference_read(), Vec::new(), vec![inputs[0]], None).unwrap()[0];
                let vary = ArrayOperation::ParallelVary(ParallelVaryOperation::new("x".to_string()));
                let value = builder.add_instruction(vary, Vec::new(), vec![value], None).unwrap()[0];
                let add = ArrayOperation::Add(AddOperation::new());
                builder.add_instruction(add, Vec::new(), vec![value, inputs[1]], None).unwrap().to_vec()
            },
        );
        let input_types = vec![
            ArrayIrType::Reference(ReferenceType::new(f32_type(&[2], None))),
            ArrayIrType::Array(f32_type(&[4], None)),
        ];
        let program = shard_map_program(body, input_types, shard_map);
        let reference = ArrayReference::new(Array::from_elements(f32_type(&[2], None), &[10.0f32, 20.0]).unwrap());
        assert_eq!(
            program.interpret(vec![
                ArrayIrValue::Reference(reference.clone()),
                f32_value(f32_type(&[4], None), &[1.0, 2.0, 3.0, 4.0]),
            ]),
            Ok(vec![f32_value(f32_type(&[4], Some(sharded)), &[11.0, 22.0, 13.0, 24.0])]),
        );
        assert_eq!(reference.read(), Ok(Array::from_elements(f32_type(&[2], None), &[10.0f32, 20.0]).unwrap()));
    }

    #[test]
    fn test_shard_map_interpretation_mutated_replicated_references() {
        // A reference that is sharded along `y` and replicated along `x` gives each device the shard of its coordinate
        // along `y`, and the devices along `x` hold identical copies of that shard. Doubling every copy keeps them
        // identical, so the caller's reference receives the doubled shards of the devices at coordinate zero along `x`.
        let mesh = manual_mesh_2x2();
        let sharded = sharding(&mesh, vec![ShardingDimension::sharded(["y"])]);
        let shard_map = ShardMap::new(mesh, vec![sharded.clone()], Vec::new(), Vec::new()).unwrap();
        let local_type = shard_map.local_input_type(0, &f32_type(&[4], Some(sharded))).unwrap();
        let body = build_program(vec![ArrayIrType::Reference(ReferenceType::new(local_type))], |builder, inputs| {
            let value = builder.add_instruction(reference_read(), Vec::new(), vec![inputs[0]], None).unwrap()[0];
            let add_update = ReferenceAddUpdateOperation::<ArrayType, ArrayIrType, ArrayReferenceTransform>::new();
            builder.add_instruction(add_update, Vec::new(), vec![inputs[0], value], None).unwrap();
            Vec::new()
        });
        let global_type = f32_type(&[4], None);
        let program =
            shard_map_program(body, vec![ArrayIrType::Reference(ReferenceType::new(global_type.clone()))], shard_map);
        let reference =
            ArrayReference::new(Array::from_elements(global_type.clone(), &[1.0f32, 2.0, 3.0, 4.0]).unwrap());
        assert_eq!(program.interpret(vec![ArrayIrValue::Reference(reference.clone())]), Ok(Vec::new()));
        assert_eq!(reference.read(), Ok(Array::from_elements(global_type, &[2.0f32, 4.0, 6.0, 8.0]).unwrap()));
    }

    #[test]
    fn test_shard_map_interpretation_unreduced_references() {
        // A reference input that is unreduced along `x` holds its pending sum, which device 0 receives while device 1
        // receives zeros, and the final partial states of both devices are summed into the caller's reference, as the
        // final-state output of the discharged map, whose output sharding is the input sharding, sums them. A body
        // that only reads the reference therefore leaves it unchanged, and doubling the partial state of each device
        // doubles the pending sum.
        let mesh = manual_mesh();
        let unreduced = Sharding::replicated(mesh.clone(), 1).with_unreduced_axes(["x"]).unwrap();
        let unreduced_type = f32_type(&[2], Some(unreduced.clone()));
        let shard_map = ShardMap::new(mesh, vec![unreduced.clone()], vec![unreduced], Vec::new()).unwrap();
        let local_type = shard_map.local_input_type(0, &unreduced_type).unwrap();
        let program = |mutate: bool| {
            let body = build_program(
                vec![ArrayIrType::Reference(ReferenceType::new(local_type.clone()))],
                |builder, inputs| {
                    if mutate {
                        let value =
                            builder.add_instruction(reference_read(), Vec::new(), vec![inputs[0]], None).unwrap()[0];
                        let add_update =
                            ReferenceAddUpdateOperation::<ArrayType, ArrayIrType, ArrayReferenceTransform>::new();
                        builder.add_instruction(add_update, Vec::new(), vec![inputs[0], value], None).unwrap();
                    }
                    builder.add_instruction(reference_read(), Vec::new(), vec![inputs[0]], None).unwrap().to_vec()
                },
            );
            let input_type = ArrayIrType::Reference(ReferenceType::new(unreduced_type.clone()));
            shard_map_program(body, vec![input_type], shard_map.clone())
        };
        let reference = ArrayReference::new(Array::from_elements(unreduced_type.clone(), &[1.0f32, 2.0]).unwrap());
        assert_eq!(
            program(false).interpret(vec![ArrayIrValue::Reference(reference.clone())]),
            Ok(vec![f32_value(unreduced_type.clone(), &[1.0, 2.0])]),
        );
        assert_eq!(reference.read(), Ok(Array::from_elements(unreduced_type.clone(), &[1.0f32, 2.0]).unwrap()));
        assert_eq!(
            program(true).interpret(vec![ArrayIrValue::Reference(reference.clone())]),
            Ok(vec![f32_value(unreduced_type.clone(), &[2.0, 4.0])]),
        );
        assert_eq!(reference.read(), Ok(Array::from_elements(unreduced_type, &[2.0f32, 4.0]).unwrap()));
    }

    #[test]
    fn test_shard_map_interpretation_print() {
        // Standard error capture is not available here, so a driver that records every bound `print` observes the
        // effect order instead: each `print` of the body runs once per device, in device order, on that device's local
        // value, and the prints of one device follow the program order of the body.
        struct PrintRecordingDriver {
            /// Body of the interpreted `shard_map`.
            body: TestProgram,

            /// Label and input of every bound `print`, in binding order.
            prints: RefCell<Vec<(String, TestValue)>>,
        }

        impl RegionDriver<TestValue, TestOperation> for PrintRecordingDriver {
            fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'r, TestValue, TestOperation>>
            where
                TestValue: 'r,
                TestOperation: 'r,
            {
                std::iter::once(self.body.entry_region_ref())
            }
        }

        impl InterpretationDriver<EagerContext<TestValue, TestOperation>> for PrintRecordingDriver {
            fn interpret_region(
                &self,
                context: &EagerContext<TestValue, TestOperation>,
                index: usize,
                inputs: Vec<TestValue>,
            ) -> Result<Vec<TestValue>, ProgramError> {
                self.region(index)?.interpret_in_context(context, inputs)
            }

            fn bind<R: BindingRegionDriver<TestValue, TestOperation>>(
                &self,
                context: &EagerContext<TestValue, TestOperation>,
                operation: TestOperation,
                regions: R,
                inputs: &[TestValue],
            ) -> Result<Vec<TestValue>, ProgramError> {
                if let Some(print) = operation.projected_payload::<PrintOperation<ArrayType>>() {
                    self.prints.borrow_mut().push((print.label().to_string(), inputs[0].clone()));
                }
                context.bind(operation, regions, inputs)
            }
        }

        let mesh = manual_mesh();
        let sharded = sharding(&mesh, vec![ShardingDimension::sharded(["x"])]);
        let shard_map = ShardMap::new(mesh, vec![sharded.clone()], vec![sharded.clone()], Vec::new()).unwrap();
        let local_type = shard_map.local_input_type(0, &f32_type(&[4], Some(sharded.clone()))).unwrap();
        let print =
            |label: &str| PrintOperation::<ArrayIrType>::new(label).with_effect_class(EffectClass::DeviceOrderedIo);
        let body = build_program(vec![ArrayIrType::Array(local_type.clone())], |builder, inputs| {
            let value = builder.add_instruction(print("a"), Vec::new(), vec![inputs[0]], None).unwrap()[0];
            let add = ArrayOperation::Add(AddOperation::new());
            let value = builder.add_instruction(add, Vec::new(), vec![value, value], None).unwrap()[0];
            builder.add_instruction(print("b"), Vec::new(), vec![value], None).unwrap().to_vec()
        });
        let operation =
            ShardMapOperation::from_program(&body, vec![ArrayIrType::Array(f32_type(&[4], None))], shard_map).unwrap();
        let driver = PrintRecordingDriver { body, prints: RefCell::new(Vec::new()) };
        assert_eq!(
            operation.interpret(
                &EagerContext::<TestValue, TestOperation>::new(),
                &driver,
                &[f32_value(f32_type(&[4], None), &[1.0, 2.0, 3.0, 4.0])],
            ),
            Ok(vec![f32_value(f32_type(&[4], Some(sharded)), &[2.0, 4.0, 6.0, 8.0])]),
        );
        assert_eq!(
            driver.prints.into_inner(),
            vec![
                ("a".to_string(), f32_value(local_type.clone(), &[1.0, 2.0])),
                ("a".to_string(), f32_value(local_type.clone(), &[3.0, 4.0])),
                ("b".to_string(), f32_value(local_type.clone(), &[2.0, 4.0])),
                ("b".to_string(), f32_value(local_type, &[6.0, 8.0])),
            ],
        );
    }
}
