//! Contains functions that switch the [`MeshAxisType`]s of mesh axes for the duration of a function, which are
//! the Ryft analogues of JAX's [`jax.sharding.auto_axes`](https://docs.jax.dev/en/latest/jax.sharding.html) and
//! [`jax.sharding.explicit_axes`](https://docs.jax.dev/en/latest/jax.sharding.html). The type of a mesh axis decides
//! who owns the placement of values along it. Along an [`Explicit`](MeshAxisType::Explicit) axis, placement is part
//! of every value's type and each operation's sharding rule determines it, while along an [`Auto`](MeshAxisType::Auto)
//! axis, the backend compiler chooses it. Switching axes lets a function use the other mode locally (e.g., hand an
//! operation whose sharding rule is ambiguous over explicit axes to the compiler with [`auto_axes`], or type a
//! computation explicitly inside an otherwise automatically partitioned program with [`explicit_axes`]).
//!
//! # Conceptual Model
//!
//! Every Ryft [`Sharding`] carries its [`LogicalMesh`], and so the types of the mesh axes belong to each value's type
//! rather than to some ambient context. A switch therefore works on values only. It takes its mesh from the shardings
//! that it is given (i.e., the output shardings of [`auto_axes`] and the input shardings of [`explicit_axes`]), or from
//! the [`default_mesh`](MeshContext::default_mesh) of the context of its inputs when [`auto_axes`] is given none (i.e.,
//! for a function without outputs), retypes the selected axes of that mesh (refer to [`LogicalMesh::with_axis_types`]),
//! and runs the function inline between two sets of [`Reshard`]s:
//!
//! ```text
//!               entry reshards                 function                 exit reshards
//!   inputs ─────────────────────▶ switched ──────────────▶ switched ────────────────────▶ outputs
//!   over mesh                     inputs                   outputs                        over mesh
//! ```
//!
//! The entry reshards retype the inputs over the switched mesh, and the exit reshards retype the outputs back over the
//! original mesh. Both are ordinary [`ReshardOperation`](crate::ReshardOperation)s, so a switch adds no operation of
//! its own, transforms (e.g., differentiation and batching) see through it, and backends lower it as the reshards that
//! it consists of (e.g., the XLA backend lowers each reshard as a sharding constraint whose dimensions stay open over
//! auto axes, so that the compiler chooses the placement along them, as JAX does). Values that the function closes
//! over keep their types over the original mesh, so operations that require the meshes of their inputs to match reject
//! combining them with switched values (as JAX rejects them because the context mesh does not match the mesh of their
//! types), whereas values without a sharding combine with values of either mesh.
//!
//! The switch is not recorded in the context of the function, so the [`default_mesh`](MeshContext::default_mesh)
//! of a context, which a nested `shard_map` without a mesh uses, keeps the original axis types inside the function.

use std::collections::BTreeSet;

use thiserror::Error;

use crate::arrays::{ArrayMemberType, LogicalMesh, MeshAxisType, Sharding, ShardingError};
use crate::axes::MeshContext;
use crate::operations::sharding::reshard::Reshard;
use crate::parameters::{Parameter, ParameterError, ParameterPath, Parameterized, ParameterizedFamily};
use crate::programs::{ProgramError, Value};

/// Error returned by [`auto_axes`] and [`explicit_axes`]. It converts into [`ProgramError`] so that functions
/// traced by other operations (e.g., the body of a `shard_map`) can switch axes with the `?` operator. Its
/// [`Program`](AxisTypesError::Program) variant passes through unchanged, and every other variant becomes
/// [`ProgramError::custom`], from which callers can recover it with [`ProgramError::downcast_custom`]. The conversion
/// from [`ProgramError`] is its inverse on custom errors. A [`ProgramError::Custom`] that holds an [`AxisTypesError`]
/// converts back into that [`AxisTypesError`], and every other [`ProgramError`] becomes the
/// [`Program`](AxisTypesError::Program) variant. The two conversions therefore form a normalizing cycle,
/// so an error of a nested switch is reported as itself rather than wrapped in a [`ProgramError`].
#[derive(Clone, Debug, PartialEq, Eq, Hash, Error)]
pub enum AxisTypesError {
    #[error(transparent)]
    Program(ProgramError),

    #[error(transparent)]
    Parameter(#[from] ParameterError),

    #[error(transparent)]
    Sharding(#[from] ShardingError),

    #[error(
        "switching mesh axes to `{axis_type}` requires at least one sharding or a default mesh of the context of its \
         inputs to take the mesh from"
    )]
    MissingMesh { axis_type: MeshAxisType },

    #[error(
        "the shardings of a switch to `{axis_type}` mesh axes must share one mesh, but got `{expected}` and `{actual}`"
    )]
    MeshMismatch { axis_type: MeshAxisType, expected: LogicalMesh, actual: LogicalMesh },

    #[error("cannot switch manual mesh axis `{axis_name}` to `{axis_type}`")]
    ManualAxis { axis_type: MeshAxisType, axis_name: String },

    #[error(
        "input #{input_index} is placed over mesh `{actual}`, which is neither the mesh `{expected}` of the switch to \
         `{axis_type}` mesh axes nor that mesh with the switched axes retyped"
    )]
    InputMeshMismatch { axis_type: MeshAxisType, input_index: usize, expected: LogicalMesh, actual: LogicalMesh },

    #[error(
        "output #{output_index} is placed over mesh `{actual}`, which is neither the mesh `{expected}` of the switch \
         to `{axis_type}` mesh axes nor that mesh with the switched axes retyped"
    )]
    OutputMeshMismatch { axis_type: MeshAxisType, output_index: usize, expected: LogicalMesh, actual: LogicalMesh },

    #[error(
        "input #{input_index} carries reduction state along mesh axis `{axis_name}`, which the switch makes `auto`, \
         so it must be reduced before the switch"
    )]
    InputReductionState { input_index: usize, axis_name: String },

    #[error(
        "output #{output_index} carries reduction state along mesh axis `{axis_name}`, which is `auto` outside of \
         the switch, so it must be reduced before the switch ends"
    )]
    OutputReductionState { output_index: usize, axis_name: String },
}

impl From<ProgramError> for AxisTypesError {
    #[inline]
    fn from(error: ProgramError) -> Self {
        error.downcast_custom::<AxisTypesError>().cloned().unwrap_or(AxisTypesError::Program(error))
    }
}

impl From<AxisTypesError> for ProgramError {
    #[inline]
    fn from(error: AxisTypesError) -> Self {
        match error {
            AxisTypesError::Program(error) => error,
            error => ProgramError::custom(error),
        }
    }
}

/// Runs `function` on `inputs` with the mesh axes `axes` switched to [`Auto`](MeshAxisType::Auto), so that the backend
/// compiler rather than the type system decides the placement of the values of `function` along them, and reshards
/// its outputs to `output_shardings`. This is the analogue of
/// [`jax.sharding.auto_axes`](https://docs.jax.dev/en/latest/jax.sharding.html),
/// whose `out_sharding` it takes as `output_shardings`.
///
/// The switch is defined over the mesh of `output_shardings`, which must all share one mesh, or over the default mesh
/// of the context of `inputs` when `output_shardings` has no leaves (refer to [`MeshContext::default_mesh`]), and
/// `axes` of [`None`] selects every axis of that mesh. Each sharded input is resharded onto the switched mesh with its
/// placement along [`Auto`](MeshAxisType::Auto) axes (i.e., the switched axes as well as the axes that are already
/// auto) removed, `function` runs inline on the resharded inputs, and each output is resharded to its output sharding.
/// Inputs without a sharding pass through unchanged. When the selected axes are exactly the auto axes of the mesh,
/// nothing is switched and `function` runs directly on `inputs`, without resharding its outputs (as in JAX, which also
/// skips the switch only in that case).
///
/// The inputs and outputs may be concrete arrays (e.g., the arrays of the XLA backend, which then reshard eagerly)
/// as well as tracers, since a switch only reshards them. The function is not traced separately, so transforms apply
/// to the reshards and to the operations of `function` alike, and the conceptual model of the
/// [`axis_types`](mod@crate::operations::sharding::axis_types) module documentation explains
/// how values that `function` closes over behave.
///
/// # Example
///
/// ```rust
/// # use indoc::indoc;
/// # use pretty_assertions::assert_eq;
/// # use ryft_core::{
/// #     Array, ArrayOperation, ArrayType, DataType, LogicalMesh, MeshAxis, MeshAxisType, Sharding,
/// #     ShardingDimension, Sin, TracingContext, auto_axes,
/// # };
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mesh = LogicalMesh::new(vec![
///     MeshAxis::new("x", 2, MeshAxisType::Explicit)?,
///     MeshAxis::new("y", 2, MeshAxisType::Explicit)?,
/// ])?;
/// let input_sharding =
///     Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"])])?;
/// let input_type = ArrayType::new_static(DataType::F32, [4, 2]).with_sharding(input_sharding)?;
/// let output_sharding =
///     Sharding::new(mesh, vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])])?;
/// let (output_type, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
///     |input| Ok(auto_axes(|input| input.sin(), input, Some(&["x"]), output_sharding.clone())?),
///     input_type,
/// )?;
/// assert_eq!(output_type.sharding(), Some(&output_sharding));
/// assert_eq!(
///     program.to_string(),
///     indoc! {"
///         lambda %0:f32[4, 2][sharding={mesh<['x'=2:explicit, 'y'=2:explicit]>, [{'x'}, {'y'}]}] .
///         let %1:f32[4, 2][sharding={mesh<['x'=2:auto, 'y'=2:explicit]>, [{}, {'y'}]}] = reshard \
///                 [sharding={mesh<['x'=2:auto, 'y'=2:explicit]>, [{}, {'y'}]}] %0
///             %2:f32[4, 2][sharding={mesh<['x'=2:auto, 'y'=2:explicit]>, [{}, {'y'}]}] = sin %1
///             %3:f32[4, 2][sharding={mesh<['x'=2:explicit, 'y'=2:explicit]>, [{}, {'x'}]}] = reshard \
///                 [sharding={mesh<['x'=2:explicit, 'y'=2:explicit]>, [{}, {'x'}]}] %2
///         in (%3)
///     "}
///     .trim_end(),
/// );
/// # Ok(())
/// # }
/// ```
///
/// Inside the switch, `x` is auto, so the input loses its placement along `x` and the backend compiler decides how to
/// place the sine along it, whereas the placement along `y` stays explicit. The output is then resharded explicitly.
///
/// # Parameters
///
///   - `function`: Function that runs on the switched inputs.
///   - `inputs`: Structured input values, whose leaves are arrays.
///   - `axes`: Names of the mesh axes to switch, or [`None`] for every axis of the mesh.
///   - `output_shardings`: Shardings over the original mesh that the outputs of `function` are resharded to, with
///     the same structure as those outputs.
///
/// # Errors
///
/// Returns [`AxisTypesError::MissingMesh`] when `output_shardings` has no leaves and the context of
/// `inputs` has no default mesh, [`AxisTypesError::MeshMismatch`] when its leaves are over different meshes,
/// [`AxisTypesError::Sharding`] when `axes` names an axis that the mesh does not have, [`AxisTypesError::ManualAxis`]
/// when `axes` selects a [`Manual`](MeshAxisType::Manual) axis (which includes `axes` of [`None`] inside a manual
/// region), [`AxisTypesError::InputMeshMismatch`] or [`AxisTypesError::OutputMeshMismatch`] for an input or an output
/// of `function` placed over a mesh that is neither the mesh of the switch nor that mesh with the switched axes retyped
/// (e.g., the mesh of a `shard_map` body when the output shardings are over the mesh outside of it),
/// [`AxisTypesError::InputReductionState`] for an input that is unreduced or reduced along a switched axis (which JAX
/// would reduce implicitly, but its lowering of that reduction fails), [`AxisTypesError::Parameter`] when the outputs
/// of `function` do not have the structure of `output_shardings`, and [`AxisTypesError::Program`] for the errors
/// of `function`, of non-array leaves, and of the reshards (e.g., an output sharding that names an
/// [`Auto`](MeshAxisType::Auto) axis or that does not restate the reduction state of its output).
pub fn auto_axes<V, Input, Output, F>(
    function: F,
    inputs: Input,
    axes: Option<&[&str]>,
    output_shardings: Output::To<Sharding>,
) -> Result<Output, AxisTypesError>
where
    V: Value<Type: ArrayMemberType, Domain: MeshContext> + Reshard,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
    Output::Family: ParameterizedFamily<Sharding>,
    F: FnOnce(Input) -> Result<Output, ProgramError>,
{
    let default_mesh = || inputs.parameters().next().and_then(|input| input.domain().default_mesh());
    let switch = AxisTypeSwitch::new(output_shardings.parameters(), default_mesh, axes, MeshAxisType::Auto)?;

    // Inputs must be placed over the mesh of the switch even when nothing is switched, which rejects, for example,
    // output shardings over a mesh that already types the selected axes as auto while the inputs do not.
    for (input_index, input) in inputs.parameters().enumerate() {
        if let Some(sharding) = input.r#type().as_array_type().map_err(ProgramError::from)?.sharding() {
            switch.validate_mesh(sharding, |expected, actual| AxisTypesError::InputMeshMismatch {
                axis_type: MeshAxisType::Auto,
                input_index,
                expected,
                actual,
            })?;
        }
    }

    // Selecting exactly the axes that are already auto switches nothing.
    let auto_axes = switch
        .mesh
        .axes()
        .iter()
        .filter(|axis| axis.r#type() == MeshAxisType::Auto)
        .map(|axis| axis.name().to_string())
        .collect::<BTreeSet<_>>();
    if switch.axis_names == auto_axes {
        return Ok(function(inputs)?);
    }

    // Each sharded input loses its placement along the auto axes of the switched mesh. Its manual variation and manual
    // reduction state are carried by the reshard itself, and reduction state along a switched axis cannot be retyped,
    // since an auto axis cannot carry reduction state.
    let structure = inputs.parameter_structure();
    let inputs = inputs
        .into_parameters()
        .enumerate()
        .map(|(input_index, input)| {
            let target = match input.r#type().as_array_type().map_err(ProgramError::from)?.sharding() {
                None => None,
                Some(sharding) => {
                    if let Some(axis_name) = sharding
                        .unreduced_axes()
                        .iter()
                        .chain(sharding.reduced_axes())
                        .find(|axis_name| switch.axis_names.contains(*axis_name))
                    {
                        return Err(AxisTypesError::InputReductionState { input_index, axis_name: axis_name.clone() });
                    }
                    Some(reshard_target(sharding, &switch.switched_mesh)?)
                }
            };
            match target {
                None => Ok(input),
                Some(target) => Ok(input.reshard(&target)?),
            }
        })
        .collect::<Result<Vec<_>, AxisTypesError>>()?;
    let outputs = function(Input::from_parameters(structure, inputs)?)?;

    validate_parameter_structure(&output_shardings.parameter_paths().collect::<Vec<_>>(), &outputs)?;
    let structure = outputs.parameter_structure();
    let outputs = outputs
        .into_parameters()
        .zip(output_shardings.into_parameters())
        .enumerate()
        .map(|(output_index, (output, sharding))| {
            if let Some(output_sharding) = output.r#type().as_array_type().map_err(ProgramError::from)?.sharding() {
                switch.validate_mesh(output_sharding, |expected, actual| AxisTypesError::OutputMeshMismatch {
                    axis_type: MeshAxisType::Auto,
                    output_index,
                    expected,
                    actual,
                })?;
            }
            Ok(output.reshard(&sharding)?)
        })
        .collect::<Result<Vec<_>, AxisTypesError>>()?;
    Ok(Output::from_parameters(structure, outputs)?)
}

/// Runs `function` on `inputs` with the mesh axes `axes` switched to [`Explicit`](MeshAxisType::Explicit), so that
/// the type system rather than the backend compiler tracks the placement of the values of `function` along them,
/// after resharding the inputs to `input_shardings`. This is the analogue of
/// [`jax.sharding.explicit_axes`](https://docs.jax.dev/en/latest/jax.sharding.html),
/// whose `in_sharding` it takes as `input_shardings`.
///
/// The switch is defined over the mesh of `input_shardings`, which must all share one mesh (i.e., the mesh of the
/// inputs, whose selected axes the switch makes explicit), and `axes` of [`None`] selects every axis of that mesh.
/// Each input is resharded to its input sharding re-expressed over the switched mesh, `function` runs inline on the
/// resharded inputs, and each sharded output is resharded back onto the original mesh with its placement along the
/// axes that are [`Auto`](MeshAxisType::Auto) there removed. Outputs without a sharding pass through unchanged.
///
/// The inputs and outputs may be concrete arrays (e.g., the arrays of the XLA backend, which then reshard eagerly)
/// as well as tracers, since a switch only reshards them. The function is not traced separately, so transforms apply
/// to the reshards and to the operations of `function` alike, and the conceptual model of the
/// [`axis_types`](mod@crate::operations::sharding::axis_types) module explains how values
/// that `function` closes over behave.
///
/// # Parameters
///
///   - `function`: Function that runs on the switched inputs.
///   - `inputs`: Structured input values, whose leaves are arrays.
///   - `axes`: Names of the mesh axes to switch, or [`None`] for every axis of the mesh.
///   - `input_shardings`: Shardings over the original mesh that the inputs are resharded to once the selected axes are
///     explicit, with the same structure as `inputs`. They may therefore name the selected axes even though those axes
///     are auto in the original mesh.
///
/// # Errors
///
/// Returns [`AxisTypesError::MissingMesh`] when `input_shardings` has no leaves, [`AxisTypesError::MeshMismatch`] when
/// its leaves are over different meshes, [`AxisTypesError::Parameter`] when it does not have the structure of `inputs`,
/// [`AxisTypesError::Sharding`] when `axes` names an axis that the mesh does not have, [`AxisTypesError::ManualAxis`]
/// when `axes` selects a [`Manual`](MeshAxisType::Manual) axis (which includes `axes` of [`None`] inside a manual
/// region), [`AxisTypesError::InputMeshMismatch`] or [`AxisTypesError::OutputMeshMismatch`] for an input or output
/// placed over a mesh that is neither the mesh of the switch nor that mesh with the switched axes retyped,
/// [`AxisTypesError::OutputReductionState`] for an output that is unreduced or reduced along an axis that is auto
/// outside of the switch, and [`AxisTypesError::Program`] for the errors of `function`, of non-array leaves, and of the
/// reshards (e.g., an input sharding that names an axis that stays auto or that does not restate the reduction state of
/// its input).
pub fn explicit_axes<V, Input, Output, F>(
    function: F,
    inputs: Input,
    axes: Option<&[&str]>,
    input_shardings: Input::To<Sharding>,
) -> Result<Output, AxisTypesError>
where
    V: Value<Type: ArrayMemberType> + Reshard,
    Input: Parameterized<V>,
    Input::Family: ParameterizedFamily<Sharding>,
    Output: Parameterized<V>,
    F: FnOnce(Input) -> Result<Output, ProgramError>,
{
    // The input shardings have the structure of the inputs, so there is no context to take a default mesh from when
    // there are none.
    let switch = AxisTypeSwitch::new(input_shardings.parameters(), || None, axes, MeshAxisType::Explicit)?;
    validate_parameter_structure(&inputs.parameter_paths().collect::<Vec<_>>(), &input_shardings)?;

    // Each input is resharded to its input sharding over the switched mesh, which may name the switched axes.
    let structure = inputs.parameter_structure();
    let inputs = inputs
        .into_parameters()
        .zip(input_shardings.into_parameters())
        .enumerate()
        .map(|(input_index, (input, sharding))| {
            if let Some(input_sharding) = input.r#type().as_array_type().map_err(ProgramError::from)?.sharding() {
                switch.validate_mesh(input_sharding, |expected, actual| AxisTypesError::InputMeshMismatch {
                    axis_type: MeshAxisType::Explicit,
                    input_index,
                    expected,
                    actual,
                })?;
            }
            Ok(input.reshard(&sharding.with_mesh(switch.switched_mesh.clone())?)?)
        })
        .collect::<Result<Vec<_>, AxisTypesError>>()?;
    let outputs = function(Input::from_parameters(structure, inputs)?)?;

    // Each sharded output loses its placement along the axes that are auto in the original mesh, which cannot carry
    // reduction state, while its manual variation and manual reduction state are carried by the reshard itself.
    let structure = outputs.parameter_structure();
    let outputs = outputs
        .into_parameters()
        .enumerate()
        .map(|(output_index, output)| {
            let target = match output.r#type().as_array_type().map_err(ProgramError::from)?.sharding() {
                None => None,
                Some(sharding) => {
                    switch.validate_mesh(sharding, |expected, actual| AxisTypesError::OutputMeshMismatch {
                        axis_type: MeshAxisType::Explicit,
                        output_index,
                        expected,
                        actual,
                    })?;
                    if let Some(axis_name) = sharding
                        .unreduced_axes()
                        .iter()
                        .chain(sharding.reduced_axes())
                        .find(|axis_name| switch.mesh.axis_type(axis_name) == Some(MeshAxisType::Auto))
                    {
                        return Err(AxisTypesError::OutputReductionState {
                            output_index,
                            axis_name: axis_name.clone(),
                        });
                    }
                    Some(reshard_target(sharding, &switch.mesh)?)
                }
            };
            match target {
                None => Ok(output),
                Some(target) => Ok(output.reshard(&target)?),
            }
        })
        .collect::<Result<Vec<_>, AxisTypesError>>()?;
    Ok(Output::from_parameters(structure, outputs)?)
}

/// Mesh axis type switch of [`auto_axes`] or [`explicit_axes`].
struct AxisTypeSwitch {
    /// Mesh of the shardings that the switch is given, over which the values outside of the switch are typed.
    mesh: LogicalMesh,

    /// [`Self::mesh`] with the switched axes retyped, over which the values inside of the switch are typed.
    switched_mesh: LogicalMesh,

    /// Names of the switched axes.
    axis_names: BTreeSet<String>,
}

impl AxisTypeSwitch {
    /// Creates the switch of the axes `axes` (or of every axis when [`None`]) to `axis_type` over the common mesh
    /// of `shardings`, or over the mesh that `default_mesh` returns when there are no shardings.
    fn new<'o>(
        shardings: impl Iterator<Item = &'o Sharding>,
        default_mesh: impl FnOnce() -> Option<LogicalMesh>,
        axes: Option<&[&str]>,
        axis_type: MeshAxisType,
    ) -> Result<Self, AxisTypesError> {
        let mut mesh = None::<&LogicalMesh>;
        for sharding in shardings {
            match mesh {
                None => mesh = Some(sharding.mesh()),
                Some(mesh) if mesh != sharding.mesh() => {
                    return Err(AxisTypesError::MeshMismatch {
                        axis_type,
                        expected: mesh.clone(),
                        actual: sharding.mesh().clone(),
                    });
                }
                Some(_) => {}
            }
        }
        let mesh = mesh.cloned().or_else(default_mesh).ok_or(AxisTypesError::MissingMesh { axis_type })?;
        let axis_names = match axes {
            Some(axes) => axes.iter().map(ToString::to_string).collect::<BTreeSet<_>>(),
            None => mesh.axes().iter().map(|axis| axis.name().to_string()).collect(),
        };
        let switched_mesh = mesh.with_axis_types(&axis_names, axis_type)?;
        if let Some(axis_name) = mesh
            .axes()
            .iter()
            .find(|axis| axis.r#type() == MeshAxisType::Manual && axis_names.contains(axis.name()))
        {
            return Err(AxisTypesError::ManualAxis { axis_type, axis_name: axis_name.name().to_string() });
        }
        Ok(Self { mesh, switched_mesh, axis_names })
    }

    /// Validates that `sharding` is over [`Self::mesh`] or [`Self::switched_mesh`], returning the error that `mismatch`
    /// builds from the expected and the actual mesh otherwise. Any other mesh would let the switch retype axes that
    /// the values type differently (e.g., axes that are manual inside a `shard_map` body whose values are switched
    /// with shardings over the mesh outside of it).
    fn validate_mesh(
        &self,
        sharding: &Sharding,
        mismatch: impl FnOnce(LogicalMesh, LogicalMesh) -> AxisTypesError,
    ) -> Result<(), AxisTypesError> {
        if sharding.mesh() == &self.mesh || sharding.mesh() == &self.switched_mesh {
            Ok(())
        } else {
            Err(mismatch(self.mesh.clone(), sharding.mesh().clone()))
        }
    }
}

/// Returns the [`Reshard`] target that retypes a value placed by `sharding` over `mesh`, which keeps its placement and
/// reduction state along the axes that are not [`Auto`](MeshAxisType::Auto) in `mesh` and leaves its manual variation
/// and manual reduction state to the reshard, which carries them over from its input.
fn reshard_target(sharding: &Sharding, mesh: &LogicalMesh) -> Result<Sharding, ShardingError> {
    let mut target = sharding.with_mesh(mesh.clone())?.without_auto_axes().without_manual_reduction_axes();
    target.clear_varying_manual_axes();
    Ok(target)
}

/// Validates that `value` has the parameter paths `expected_paths`, returning a [`ParameterError`] otherwise.
fn validate_parameter_structure<P: Parameter, S: Parameterized<P>>(
    expected_paths: &[ParameterPath],
    value: &S,
) -> Result<(), ParameterError> {
    let paths = value.parameter_paths().collect::<Vec<_>>();
    if paths != expected_paths {
        let render = |paths: &[ParameterPath]| {
            format!("[{}]", paths.iter().map(ParameterPath::to_string).collect::<Vec<_>>().join(", "))
        };
        return Err(ParameterError::MismatchedParameterStructures {
            left_structure: render(expected_paths),
            right_structure: render(paths.as_slice()),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayType, DataType, MeshAxis,
        ShardingDimension,
    };
    use crate::batching::{BatchAxis, batch};
    use crate::differentiation::{DifferentiableType, differentiate_at};
    use crate::operations::arithmetic::{Add, Mul};
    use crate::operations::collectives::parallel_reduce::ParallelReduce;
    use crate::operations::reductions::{Reduce, ReductionKind};
    use crate::operations::sharding::shard_map::{ShardMapOptions, ShardMapTracer, shard_map};
    use crate::operations::trigonometric::{Cos, Sin};
    use crate::programs::{ProjectedValue, TypeError, Typed, ValueProjection};
    use crate::tracing::{Tracer, TracingContext};

    use super::*;

    /// Error message of a `reshard` whose target names an auto mesh axis.
    const RESHARD_AUTO_TARGET_ERROR: &str =
        "`reshard` cannot target auto mesh axes; use `constrain_sharding` to constrain placement over them";

    /// Tracing context over homogeneous arrays.
    type ArrayTracingContext = TracingContext<Array, ArrayOperation<Array>>;

    /// Tracing context over the array IR test family, in which `shard_map` is invoked.
    type ArrayIrTracingContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

    /// Four-device mesh whose axes `x` and `y` are both explicit.
    fn explicit_mesh() -> LogicalMesh {
        LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap()
    }

    /// [`explicit_mesh`] with both of its axes auto.
    fn auto_mesh() -> LogicalMesh {
        explicit_mesh().with_axis_types(["x", "y"], MeshAxisType::Auto).unwrap()
    }

    /// Sharding over `mesh` that places the dimensions of a value as `dimensions` describes, where each entry names
    /// the axis that places its dimension or is [`None`] for a replicated dimension.
    fn sharding(mesh: &LogicalMesh, dimensions: &[Option<&str>]) -> Sharding {
        let dimensions = dimensions
            .iter()
            .map(|axis| match axis {
                Some(axis) => ShardingDimension::sharded([*axis]),
                None => ShardingDimension::replicated(),
            })
            .collect();
        Sharding::new(mesh.clone(), dimensions).unwrap()
    }

    /// `f32[4, 2]` type placed by `sharding` when one is provided.
    fn matrix_type(sharding: Option<Sharding>) -> ArrayType {
        ArrayType::new_static(DataType::F32, [4, 2]).with_sharding(sharding).unwrap()
    }

    /// `f32[4, 2]` array of type [`matrix_type`] holding `0, 1, ..., 7` in row-major order, which is the `data` input
    /// of the JAX oracles in `.tasks/shard_map_jax_parity_gaps/m9_oracles.py`.
    fn matrix(sharding: Option<Sharding>) -> Array {
        Array::from_elements(matrix_type(sharding), &[0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap()
    }

    /// Asserts that `actual` matches the JAX oracle values `expected` up to the rounding of `f32` transcendental
    /// functions, which may differ between implementations in their last bits.
    fn assert_close(actual: &Array, expected: &[f64]) {
        let actual = actual.to_f64s();
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-5, "got {actual}, expected {expected}");
        }
    }

    #[test]
    fn test_axis_types_error_from_program_error() {
        let error = AxisTypesError::ManualAxis { axis_type: MeshAxisType::Auto, axis_name: "x".to_string() };
        assert_eq!(AxisTypesError::from(ProgramError::from(error.clone())), error);
        let program_error = ProgramError::InvalidArgument { message: "invalid".to_string() };
        assert_eq!(AxisTypesError::from(program_error.clone()), AxisTypesError::Program(program_error));
    }

    #[test]
    fn test_axis_types_error_into_program_error() {
        let program_error = ProgramError::InvalidArgument { message: "invalid".to_string() };
        assert_eq!(ProgramError::from(AxisTypesError::Program(program_error.clone())), program_error);
        let error = AxisTypesError::ManualAxis { axis_type: MeshAxisType::Auto, axis_name: "x".to_string() };
        let program_error = ProgramError::from(error.clone());
        assert_eq!(program_error.downcast_custom::<AxisTypesError>(), Some(&error));
        assert_eq!(error.to_string(), "cannot switch manual mesh axis `x` to `auto`");
    }

    #[test]
    fn test_auto_axes() {
        // The body sees `x` as auto, so its input loses its placement along `x`, and the reduction over the dimension
        // that `y` places stays typed explicitly. The output is resharded along `x` over the original mesh, as in the
        // JAX oracle `auto_axes(x)`, whose jaxpr has the same two reshards.
        let mesh = explicit_mesh();
        let input_sharding = sharding(&mesh, &[Some("x"), Some("y")]);
        let output_sharding = sharding(&mesh, &[Some("x")]);
        let function = |input: Tracer<ArrayTracingContext>| {
            auto_axes(
                |input: Tracer<ArrayTracingContext>| {
                    let sine = input.sin()?;
                    sine.add(&sine)?.reduce_sum(&[1], None)
                },
                input,
                Some(&["x"]),
                output_sharding.clone(),
            )
        };
        let (output_type, program) =
            ArrayTracingContext::trace(|input| Ok(function(input)?), matrix_type(Some(input_sharding.clone())))
                .unwrap();
        assert_eq!(output_type.sharding(), Some(&output_sharding));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[4, 2][sharding={mesh<['x'=2:explicit, 'y'=2:explicit]>, [{'x'}, {'y'}]}] .
                let %1:f32[4, 2][sharding={mesh<['x'=2:auto, 'y'=2:explicit]>, [{}, {'y'}]}] = reshard \
                        [sharding={mesh<['x'=2:auto, 'y'=2:explicit]>, [{}, {'y'}]}] %0
                    %2:f32[4, 2][sharding={mesh<['x'=2:auto, 'y'=2:explicit]>, [{}, {'y'}]}] = sin %1
                    %3:f32[4, 2][sharding={mesh<['x'=2:auto, 'y'=2:explicit]>, [{}, {'y'}]}] = add %2 %2
                    %4:f32[4][sharding={mesh<['x'=2:auto, 'y'=2:explicit]>, [{}]}] = reduce [kind=sum, axes=[1]] %3
                    %5:f32[4][sharding={mesh<['x'=2:explicit, 'y'=2:explicit]>, [{'x'}]}] = reshard \
                        [sharding={mesh<['x'=2:explicit, 'y'=2:explicit]>, [{'x'}]}] %4
                in (%5)
            "}
            .trim_end(),
        );

        // Concrete arrays reshard eagerly, and interpreting the staged program agrees with them and with JAX.
        let output = auto_axes(
            |input: Array| {
                let sine = input.sin()?;
                sine.add(&sine)?.reduce_sum(&[1], None)
            },
            matrix(Some(input_sharding.clone())),
            Some(&["x"]),
            output_sharding.clone(),
        )
        .unwrap();
        assert_eq!(output.r#type().sharding(), Some(&output_sharding));
        assert_close(&output, &[1.6829419136047363, 2.100834846496582, -3.4314537048339844, 0.7551422119140625]);
        assert_eq!(program.interpret(matrix(Some(input_sharding))), Ok(output));
    }

    #[test]
    fn test_auto_axes_selects_every_axis_for_none() {
        // Without axes, every axis is switched, so the body input is replicated over an all-auto mesh (JAX's
        // `float32[4,2]`), and the output is resharded to an explicit placement along both axes.
        let mesh = explicit_mesh();
        let input_sharding = sharding(&mesh, &[Some("x"), Some("y")]);
        let output_sharding = sharding(&mesh, &[Some("y"), Some("x")]);
        let (_, program) = ArrayTracingContext::trace(
            |input| {
                Ok(auto_axes(
                    |input: Tracer<ArrayTracingContext>| {
                        assert_eq!(input.r#type().sharding(), Some(&Sharding::replicated(auto_mesh(), 2)));
                        input.add(&input)
                    },
                    input,
                    None,
                    output_sharding.clone(),
                )?)
            },
            matrix_type(Some(input_sharding.clone())),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[4, 2][sharding={mesh<['x'=2:explicit, 'y'=2:explicit]>, [{'x'}, {'y'}]}] .
                let %1:f32[4, 2][sharding={mesh<['x'=2:auto, 'y'=2:auto]>, [{}, {}]}] = reshard \
                        [sharding={mesh<['x'=2:auto, 'y'=2:auto]>, [{}, {}]}] %0
                    %2:f32[4, 2][sharding={mesh<['x'=2:auto, 'y'=2:auto]>, [{}, {}]}] = add %1 %1
                    %3:f32[4, 2][sharding={mesh<['x'=2:explicit, 'y'=2:explicit]>, [{'y'}, {'x'}]}] = reshard \
                        [sharding={mesh<['x'=2:explicit, 'y'=2:explicit]>, [{'y'}, {'x'}]}] %2
                in (%3)
            "}
            .trim_end(),
        );
        assert_eq!(
            program.interpret(matrix(Some(input_sharding))).map(|output| output.to_f64s()),
            Ok(vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]),
        );
    }

    #[test]
    fn test_auto_axes_switches_nothing_when_selecting_exactly_the_auto_axes() {
        // Selecting exactly the auto axes of the mesh runs the function directly and leaves its outputs as they are,
        // ignoring the output sharding, as JAX's `auto_axes(y)` over a mesh whose only auto axis is `y` does.
        let mesh = explicit_mesh().with_axis_types(["y"], MeshAxisType::Auto).unwrap();
        let input_sharding = sharding(&mesh, &[Some("x"), None]);
        let replicated = Sharding::replicated(mesh.clone(), 2);
        let output = auto_axes(
            |input: Array| input.add(&input),
            matrix(Some(input_sharding.clone())),
            Some(&["y"]),
            replicated.clone(),
        )
        .unwrap();
        assert_eq!(output.r#type().sharding(), Some(&input_sharding));

        // Selecting more axes switches them all and reshards the output, as JAX's `auto_axes(x, y)` does.
        let output = auto_axes(
            |input: Array| input.add(&input),
            matrix(Some(input_sharding)),
            Some(&["x", "y"]),
            replicated.clone(),
        )
        .unwrap();
        assert_eq!(output.r#type().sharding(), Some(&replicated));
        assert_eq!(output.to_f64s(), vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]);
    }

    #[test]
    fn test_auto_axes_passes_unsharded_inputs_through() {
        let output_sharding = sharding(&explicit_mesh(), &[Some("x"), None]);
        let output = auto_axes(|input: Array| Ok(input), matrix(None), Some(&["x"]), output_sharding.clone()).unwrap();
        assert_eq!(output, matrix(Some(output_sharding)));
    }

    #[test]
    fn test_auto_axes_preserves_reduction_state_along_other_axes() {
        // An input that is unreduced along `y` keeps that state through a switch of `x`, and the output sharding
        // restates it, whereas an input that is unreduced along the switched axis is rejected.
        let mesh = explicit_mesh();
        let unreduced_along_y = Sharding::replicated(mesh.clone(), 2).with_unreduced_axes(["y"]).unwrap();
        let output = auto_axes(
            |input: Array| Ok(input),
            matrix(Some(unreduced_along_y.clone())),
            Some(&["x"]),
            unreduced_along_y.clone(),
        );
        assert_eq!(output, Ok(matrix(Some(unreduced_along_y.clone()))));
        assert_eq!(
            auto_axes(
                |input: Array| Ok(input),
                matrix(Some(unreduced_along_y.clone())),
                Some(&["y"]),
                unreduced_along_y,
            ),
            Err(AxisTypesError::InputReductionState { input_index: 0, axis_name: "y".to_string() }),
        );
        let reduced_along_x = Sharding::replicated(mesh.clone(), 2).with_reduced_axes(["x"]).unwrap();
        assert_eq!(
            auto_axes(
                |inputs: (Array, Array)| Ok(inputs),
                (matrix(None), matrix(Some(reduced_along_x))),
                None,
                (Sharding::replicated(mesh.clone(), 2), Sharding::replicated(mesh, 2)),
            ),
            Err(AxisTypesError::InputReductionState { input_index: 1, axis_name: "x".to_string() }),
        );
    }

    #[test]
    fn test_auto_axes_rejects_invalid_switches() {
        let mesh = explicit_mesh();
        let other_mesh = LogicalMesh::new(vec![MeshAxis::new("z", 4, MeshAxisType::Explicit).unwrap()]).unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 2);

        // The mesh comes from the output shardings, which must share one mesh, or, without output shardings, from the
        // default mesh of the context of the inputs, which an eager context does not have.
        assert_eq!(
            auto_axes(|_: Array| Ok(Vec::<Array>::new()), matrix(None), None, Vec::new()),
            Err(AxisTypesError::MissingMesh { axis_type: MeshAxisType::Auto }),
        );
        assert_eq!(
            auto_axes(
                |input: Array| Ok((input.clone(), input)),
                matrix(None),
                None,
                (replicated.clone(), Sharding::replicated(other_mesh.clone(), 2)),
            ),
            Err(AxisTypesError::MeshMismatch {
                axis_type: MeshAxisType::Auto,
                expected: mesh.clone(),
                actual: other_mesh.clone(),
            }),
        );

        // The selected axes must be non-manual axes of that mesh.
        assert_eq!(
            auto_axes(|input: Array| Ok(input), matrix(None), Some(&["w"]), replicated.clone()),
            Err(AxisTypesError::Sharding(ShardingError::UnknownMeshAxisName { name: "w".to_string() })),
        );
        let manual_mesh = mesh.with_axis_types(["y"], MeshAxisType::Manual).unwrap();
        assert_eq!(
            auto_axes(|input: Array| Ok(input), matrix(None), None, Sharding::replicated(manual_mesh, 2)),
            Err(AxisTypesError::ManualAxis { axis_type: MeshAxisType::Auto, axis_name: "y".to_string() }),
        );

        // Inputs must be placed over the mesh of the switch, up to the types of the switched axes.
        assert_eq!(
            auto_axes(
                |input: Array| Ok(input),
                matrix(Some(sharding(&other_mesh, &[Some("z"), None]))),
                Some(&["x"]),
                replicated.clone(),
            ),
            Err(AxisTypesError::InputMeshMismatch {
                axis_type: MeshAxisType::Auto,
                input_index: 0,
                expected: mesh.clone(),
                actual: other_mesh.clone(),
            }),
        );
        let switched_mesh = mesh.with_axis_types(["x"], MeshAxisType::Auto).unwrap();
        assert_eq!(
            auto_axes(
                |input: Array| Ok(input),
                matrix(Some(sharding(&switched_mesh, &[None, Some("y")]))),
                Some(&["x"]),
                replicated.clone(),
            ),
            Ok(matrix(Some(replicated.clone()))),
        );

        // Inputs are validated even when nothing is switched, so output shardings over a mesh that already types `x` as
        // auto cannot silently skip the switch of inputs whose mesh types `x` as explicit.
        let auto_x_mesh = mesh.with_axis_types(["x"], MeshAxisType::Auto).unwrap();
        assert_eq!(
            auto_axes(
                |input: Array| Ok(input),
                matrix(Some(sharding(&mesh, &[Some("x"), None]))),
                Some(&["x"]),
                Sharding::replicated(auto_x_mesh.clone(), 2),
            ),
            Err(AxisTypesError::InputMeshMismatch {
                axis_type: MeshAxisType::Auto,
                input_index: 0,
                expected: auto_x_mesh,
                actual: mesh.clone(),
            }),
        );

        // Outputs of the function must be placed over the mesh of the switch too.
        let unrelated = matrix(Some(Sharding::replicated(other_mesh.clone(), 2)));
        assert_eq!(
            auto_axes(|_: Array| Ok(unrelated.clone()), matrix(None), Some(&["x"]), replicated.clone()),
            Err(AxisTypesError::OutputMeshMismatch {
                axis_type: MeshAxisType::Auto,
                output_index: 0,
                expected: mesh.clone(),
                actual: other_mesh.clone(),
            }),
        );

        // The outputs must have the structure of the output shardings, and the reshards validate the shardings.
        assert_eq!(
            auto_axes(|input: Array| Ok(vec![input.clone(), input]), matrix(None), None, vec![replicated.clone()]),
            Err(AxisTypesError::Parameter(ParameterError::MismatchedParameterStructures {
                left_structure: "[$[0]]".to_string(),
                right_structure: "[$[0], $[1]]".to_string(),
            })),
        );
        let mixed_mesh = mesh.with_axis_types(["y"], MeshAxisType::Auto).unwrap();
        assert_eq!(
            auto_axes(|input: Array| Ok(input), matrix(None), Some(&["x"]), sharding(&mixed_mesh, &[Some("y"), None])),
            Err(AxisTypesError::Program(ProgramError::Type(TypeError::invalid(RESHARD_AUTO_TARGET_ERROR)))),
        );
    }

    #[test]
    fn test_auto_axes_rejects_captured_values_over_the_original_mesh() {
        // A value that the function closes over keeps its type over the original mesh, which differs from the switched
        // mesh of the body values, so combining them fails (as JAX rejects it because the context mesh does not match
        // the mesh of its type). A captured value without a sharding combines with them.
        let mesh = explicit_mesh();
        let input_sharding = sharding(&mesh, &[Some("x"), Some("y")]);
        let error = ArrayTracingContext::trace(
            |(input, captured): (Tracer<ArrayTracingContext>, Tracer<ArrayTracingContext>)| {
                Ok(auto_axes(
                    |input: Tracer<ArrayTracingContext>| input.add(&captured),
                    input,
                    Some(&["x"]),
                    input_sharding.clone(),
                )?)
            },
            (matrix_type(Some(input_sharding.clone())), matrix_type(Some(input_sharding.clone()))),
        )
        .unwrap_err();
        assert_eq!(error, ProgramError::Type(TypeError::invalid("`add` input types are not broadcast-compatible")));
        let captured = matrix(None);
        let output = auto_axes(
            |input: Array| input.add(&captured),
            matrix(Some(input_sharding.clone())),
            Some(&["x"]),
            input_sharding,
        );
        assert_eq!(output.map(|output| output.to_f64s()), Ok(vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]));
    }

    #[test]
    fn test_auto_axes_inside_partially_manual_shard_map() {
        // Inside a `shard_map` that makes `x` manual, the free axis `y` can be switched, and the body values keep their
        // manual variation along `x` (JAX's `float32[2,2]{V:x}`). Switching every axis selects the manual axis `x`,
        // which is rejected, as in JAX.
        let mesh = explicit_mesh().with_axis_types(["x"], MeshAxisType::Manual).unwrap();
        let global_sharding = sharding(&mesh, &[Some("x"), Some("y")]);
        let local_sharding = sharding(&mesh, &[None, Some("y")]);
        let trace = |axes: Option<&'static [&'static str]>, output_sharding: &Sharding| {
            ArrayIrTracingContext::trace(
                |input: Tracer<ArrayIrTracingContext>| {
                    let input = ValueProjection::<ArrayType>::into_projected(input).map_err(ProgramError::from)?;
                    let output = shard_map(
                        |input: ShardMapTracer<ArrayIrTracingContext>| {
                            Ok(auto_axes(
                                |input: ShardMapTracer<ArrayIrTracingContext>| input.add(&input),
                                input,
                                axes,
                                output_sharding.clone(),
                            )?)
                        },
                        input,
                        mesh.clone(),
                        global_sharding.clone(),
                        global_sharding.clone(),
                        ShardMapOptions::new(),
                    )?;
                    Ok(ProjectedValue::into_value(output))
                },
                ArrayIrType::Array(matrix_type(None)),
            )
        };
        let (_, program) = trace(Some(&["y"]), &local_sharding).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[4, 2] .
                let %1:f32[4, 2][sharding={mesh<['x'=2:manual, 'y'=2:explicit]>, [{'x'}, {'y'}]}] = shard_map [
                    mesh=['x'=2:manual, 'y'=2:explicit],
                    input_shardings=[{mesh<['x'=2:manual, 'y'=2:explicit]>, [{'x'}, {'y'}]}],
                    output_shardings=[{mesh<['x'=2:manual, 'y'=2:explicit]>, [{'x'}, {'y'}]}],
                    manual_axes=['x'],
                    global_input_types=[f32[4, 2][sharding={mesh<['x'=2:manual, 'y'=2:explicit]>, [{'x'}, {'y'}]}]],
                    global_output_types=[f32[4, 2][sharding={mesh<['x'=2:manual, 'y'=2:explicit]>, [{'x'}, {'y'}]}]],
                ] %0 [
                    body={
                        lambda \
                            %0:f32[2, 2][sharding={mesh<['x'=2:manual, 'y'=2:explicit]>, [{}, {'y'}], varying_manual={'x'}}] .
                        let %1:f32[2, 2][sharding={mesh<['x'=2:manual, 'y'=2:auto]>, [{}, {}], varying_manual={'x'}}] = \
                                reshard [sharding={mesh<['x'=2:manual, 'y'=2:auto]>, [{}, {}]}] %0
                            %2:f32[2, 2][sharding={mesh<['x'=2:manual, 'y'=2:auto]>, [{}, {}], varying_manual={'x'}}] = \
                                add %1 %1
                            %3:f32[2, 2][sharding={mesh<['x'=2:manual, 'y'=2:explicit]>, [{}, {'y'}], varying_manual={'x'}}] = \
                                reshard [sharding={mesh<['x'=2:manual, 'y'=2:explicit]>, [{}, {'y'}]}] %2
                        in (%3)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );
        assert_eq!(
            program.interpret(ArrayIrValue::Array(matrix(None))),
            Ok(ArrayIrValue::Array(
                Array::from_elements(
                    matrix_type(Some(global_sharding.clone())),
                    &[0.0f32, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0],
                )
                .unwrap(),
            )),
        );
        assert_eq!(
            trace(None, &local_sharding).unwrap_err().downcast_custom::<AxisTypesError>(),
            Some(&AxisTypesError::ManualAxis { axis_type: MeshAxisType::Auto, axis_name: "x".to_string() }),
        );

        // Shardings over the mesh outside of the map do not type `x` as manual, so they would retype the body values
        // along `x`, which is rejected (as JAX rejects shardings whose mesh differs from the context mesh).
        let outer_mesh = explicit_mesh();
        assert_eq!(
            trace(None, &sharding(&outer_mesh, &[None, Some("y")]))
                .unwrap_err()
                .downcast_custom::<AxisTypesError>(),
            Some(&AxisTypesError::InputMeshMismatch {
                axis_type: MeshAxisType::Auto,
                input_index: 0,
                expected: outer_mesh,
                actual: mesh,
            }),
        );
    }

    #[test]
    fn test_auto_axes_of_free_axes_keeps_collectives_over_manual_axes() {
        // Switching the free axis `y` inside a `shard_map` that makes `x` manual retypes only `y`, so the switched
        // values still belong to the manual region of `x`, and a collective over `x` accepts them and keeps them on
        // the switched mesh, as JAX's `psum` over `x` does inside `auto_axes`, which retypes its context mesh.
        let mesh = explicit_mesh().with_axis_types(["x"], MeshAxisType::Manual).unwrap();
        let input_sharding = sharding(&mesh, &[Some("x"), Some("y")]);
        let output_sharding = sharding(&mesh, &[None, Some("y")]);
        let reduced_type = RefCell::new(None);
        let (_, program) = ArrayIrTracingContext::trace(
            |input: Tracer<ArrayIrTracingContext>| {
                let input = ValueProjection::<ArrayType>::into_projected(input).map_err(ProgramError::from)?;
                let output = shard_map(
                    |input: ShardMapTracer<ArrayIrTracingContext>| {
                        Ok(auto_axes(
                            |input: ShardMapTracer<ArrayIrTracingContext>| {
                                let reduced = input.parallel_reduce(ReductionKind::Sum, "x")?;
                                *reduced_type.borrow_mut() = Some(reduced.r#type().into_owned());
                                Ok(reduced)
                            },
                            input,
                            Some(&["y"]),
                            output_sharding.clone(),
                        )?)
                    },
                    input,
                    mesh.clone(),
                    input_sharding.clone(),
                    output_sharding.clone(),
                    ShardMapOptions::new(),
                )?;
                Ok(ProjectedValue::into_value(output))
            },
            ArrayIrType::Array(matrix_type(None)),
        )
        .unwrap();
        let switched_mesh = mesh.with_axis_types(["y"], MeshAxisType::Auto).unwrap();
        assert_eq!(
            reduced_type.into_inner(),
            Some(
                ArrayType::new_static(DataType::F32, [2, 2])
                    .with_sharding(Sharding::replicated(switched_mesh, 2))
                    .unwrap()
            ),
        );
        let ArrayIrValue::Array(output) = program.interpret(ArrayIrValue::Array(matrix(None))).unwrap() else {
            panic!("expected an array output");
        };
        assert_eq!(output.to_f64s(), vec![4.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn test_auto_axes_of_shard_map_outputs() {
        // The outputs of a `shard_map` are typed over the mesh of its caller, so they can be switched directly, as in
        // JAX, rather than first being resharded off of the mesh of the map, which types `x` as manual.
        let mesh = explicit_mesh();
        let map_mesh = mesh.with_axis_types(["x"], MeshAxisType::Manual).unwrap();
        let input_sharding = sharding(&mesh, &[Some("x"), Some("y")]);
        let map_sharding = sharding(&map_mesh, &[Some("x"), Some("y")]);
        let (output_type, program) = ArrayIrTracingContext::trace(
            |input: Tracer<ArrayIrTracingContext>| {
                let input = ValueProjection::<ArrayType>::into_projected(input).map_err(ProgramError::from)?;
                let output = shard_map(
                    |input: ShardMapTracer<ArrayIrTracingContext>| Ok(input.clone() + input),
                    input,
                    map_mesh.clone(),
                    map_sharding.clone(),
                    map_sharding.clone(),
                    ShardMapOptions::new(),
                )?;
                let output = auto_axes(
                    |output: ShardMapTracer<ArrayIrTracingContext>| output.sin(),
                    output,
                    Some(&["y"]),
                    input_sharding.clone(),
                )?;
                Ok(ProjectedValue::into_value(output))
            },
            ArrayIrType::Array(matrix_type(Some(input_sharding.clone()))),
        )
        .unwrap();
        assert_eq!(output_type, ArrayIrType::Array(matrix_type(Some(input_sharding.clone()))));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[4, 2][sharding={mesh<['x'=2:explicit, 'y'=2:explicit]>, [{'x'}, {'y'}]}] .
                let %1:f32[4, 2][sharding={mesh<['x'=2:explicit, 'y'=2:explicit]>, [{'x'}, {'y'}]}] = shard_map [
                    mesh=['x'=2:manual, 'y'=2:explicit],
                    caller_mesh=['x'=2:explicit, 'y'=2:explicit],
                    input_shardings=[{mesh<['x'=2:manual, 'y'=2:explicit]>, [{'x'}, {'y'}]}],
                    output_shardings=[{mesh<['x'=2:manual, 'y'=2:explicit]>, [{'x'}, {'y'}]}],
                    manual_axes=['x'],
                    global_input_types=[f32[4, 2][sharding={mesh<['x'=2:manual, 'y'=2:explicit]>, [{'x'}, {'y'}]}]],
                    global_output_types=[f32[4, 2][sharding={mesh<['x'=2:explicit, 'y'=2:explicit]>, [{'x'}, {'y'}]}]],
                ] %0 [
                    body={
                        lambda \
                            %0:f32[2, 2][sharding={mesh<['x'=2:manual, 'y'=2:explicit]>, [{}, {'y'}], varying_manual={'x'}}] .
                        let %1:f32[2, 2][sharding={mesh<['x'=2:manual, 'y'=2:explicit]>, [{}, {'y'}], varying_manual={'x'}}] = \
                                add %0 %0
                        in (%1)
                    },
                ]
                    %2:f32[4, 2][sharding={mesh<['x'=2:explicit, 'y'=2:auto]>, [{'x'}, {}]}] = reshard \
                        [sharding={mesh<['x'=2:explicit, 'y'=2:auto]>, [{'x'}, {}]}] %1
                    %3:f32[4, 2][sharding={mesh<['x'=2:explicit, 'y'=2:auto]>, [{'x'}, {}]}] = sin %2
                    %4:f32[4, 2][sharding={mesh<['x'=2:explicit, 'y'=2:explicit]>, [{'x'}, {'y'}]}] = reshard \
                        [sharding={mesh<['x'=2:explicit, 'y'=2:explicit]>, [{'x'}, {'y'}]}] %3
                in (%4)
            "}
            .trim_end(),
        );
        let ArrayIrValue::Array(output) = program.interpret(ArrayIrValue::Array(matrix(Some(input_sharding)))).unwrap()
        else {
            panic!("expected an array output");
        };
        assert_close(
            &output,
            &[0.0, 0.9092974, -0.7568025, -0.2794155, 0.98935825, -0.54402111, -0.53657292, 0.99060736],
        );
    }

    #[test]
    fn test_auto_axes_without_outputs_takes_the_default_mesh_of_its_context() {
        // Inside a `shard_map` body, whose context has the mesh of the map as its default mesh, a switch without
        // output shardings takes its mesh from that context, as JAX's `auto_axes` uses its context mesh, while a switch
        // whose inputs have no context mesh needs output shardings (refer to `test_auto_axes_rejects_invalid_switches`).
        let mesh = explicit_mesh().with_axis_types(["x"], MeshAxisType::Manual).unwrap();
        let global_sharding = sharding(&mesh, &[Some("x"), Some("y")]);
        let switched_type = RefCell::new(None);
        ArrayIrTracingContext::trace(
            |input: Tracer<ArrayIrTracingContext>| {
                let input = ValueProjection::<ArrayType>::into_projected(input).map_err(ProgramError::from)?;
                let output = shard_map(
                    |input: ShardMapTracer<ArrayIrTracingContext>| {
                        auto_axes(
                            |input: ShardMapTracer<ArrayIrTracingContext>| {
                                *switched_type.borrow_mut() = Some(input.r#type().into_owned());
                                Ok(Vec::<ShardMapTracer<ArrayIrTracingContext>>::new())
                            },
                            input.clone(),
                            Some(&["y"]),
                            Vec::new(),
                        )?;
                        Ok(input)
                    },
                    input,
                    mesh.clone(),
                    global_sharding.clone(),
                    global_sharding.clone(),
                    ShardMapOptions::new(),
                )?;
                Ok(ProjectedValue::into_value(output))
            },
            ArrayIrType::Array(matrix_type(None)),
        )
        .unwrap();
        let switched_mesh = mesh.with_axis_types(["y"], MeshAxisType::Auto).unwrap();
        let switched_sharding = Sharding::replicated(switched_mesh, 2).with_varying_manual_axes(["x"]).unwrap();
        assert_eq!(
            switched_type.into_inner(),
            Some(ArrayType::new_static(DataType::F32, [2, 2]).with_sharding(switched_sharding).unwrap()),
        );
    }

    #[test]
    fn test_explicit_axes_inside_partially_manual_shard_map() {
        // Inside a `shard_map` that makes `x` manual, the free auto axis `y` can be made explicit, and the body values
        // keep their manual variation along `x` throughout.
        let mesh = auto_mesh().with_axis_types(["x"], MeshAxisType::Manual).unwrap();
        let global_sharding = sharding(&mesh, &[Some("x"), None]);
        let (_, program) = ArrayIrTracingContext::trace(
            |input: Tracer<ArrayIrTracingContext>| {
                let input = ValueProjection::<ArrayType>::into_projected(input).map_err(ProgramError::from)?;
                let output = shard_map(
                    |input: ShardMapTracer<ArrayIrTracingContext>| {
                        let output = explicit_axes(
                            |input: ShardMapTracer<ArrayIrTracingContext>| {
                                let switched_mesh = mesh.with_axis_types(["y"], MeshAxisType::Explicit).unwrap();
                                let expected = sharding(&switched_mesh, &[None, Some("y")])
                                    .with_varying_manual_axes(["x"])
                                    .unwrap();
                                assert_eq!(input.r#type().sharding(), Some(&expected));
                                input.add(&input)
                            },
                            input,
                            Some(&["y"]),
                            sharding(&mesh, &[None, Some("y")]),
                        )?;
                        let expected = Sharding::replicated(mesh.clone(), 2).with_varying_manual_axes(["x"]).unwrap();
                        assert_eq!(output.r#type().sharding(), Some(&expected));
                        Ok(output)
                    },
                    input,
                    mesh.clone(),
                    global_sharding.clone(),
                    global_sharding.clone(),
                    ShardMapOptions::new(),
                )?;
                Ok(ProjectedValue::into_value(output))
            },
            ArrayIrType::Array(matrix_type(None)),
        )
        .unwrap();
        assert_eq!(
            program.interpret(ArrayIrValue::Array(matrix(None))),
            Ok(ArrayIrValue::Array(
                Array::from_elements(
                    matrix_type(Some(global_sharding)),
                    &[0.0f32, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0],
                )
                .unwrap(),
            )),
        );
    }

    #[test]
    fn test_auto_axes_differentiation() {
        // Forward and reverse mode differentiate through the reshards, which retype tangents and cotangents like their
        // primals, and agree with the JAX oracles `jvp auto_axes` and `grad auto_axes`.
        let mesh = explicit_mesh();
        let input_sharding = sharding(&mesh, &[Some("x"), Some("y")]);
        let expected = [
            0.0,
            1.3817732334136963,
            0.07700371742248535,
            -2.828857421875,
            -3.3713769912719727,
            0.4593867063522339,
            5.4816060066223145,
            5.93430233001709,
        ];
        let (_, tangent) = differentiate_at(matrix(Some(input_sharding.clone())))
            .jvp(Array::from_elements(matrix_type(Some(input_sharding.clone())), &[1.0f32; 8]).unwrap(), |input| {
                Ok(auto_axes(|input| input.sin()?.mul(&input), input, Some(&["x"]), input_sharding.clone())?)
            })
            .unwrap();
        assert_eq!(tangent.r#type().sharding(), Some(&input_sharding));
        assert_close(&tangent, &expected);

        let gradient = differentiate_at(matrix(Some(input_sharding.clone())))
            .gradient(|input| {
                Ok(auto_axes(
                    |input| input.sin()?.mul(&input)?.reduce_sum(&[0, 1], None),
                    input,
                    Some(&["x"]),
                    Sharding::replicated(mesh.clone(), 0),
                )?)
            })
            .unwrap();
        assert_eq!(gradient.r#type().sharding(), Some(&input_sharding));
        assert_close(&gradient, &expected);
    }

    #[test]
    fn test_auto_axes_batching() {
        // Elementwise sine of `matrix`, which is the value of the JAX oracles `vmap auto_axes` and
        // `vmap auto_axes(x) over x-placed batch`.
        const SINE: [f64; 8] = [
            0.0,
            0.8414709568023682,
            0.9092974066734314,
            0.14112000167369843,
            -0.756802499294281,
            -0.9589242935180664,
            -0.279415488243103,
            0.6569865942001343,
        ];

        // Batching over rows placed along `x` lifts both reshards with the batch dimension placed along `x`, which
        // stays explicit when only `y` is switched, as in the JAX oracle `vmap auto_axes`.
        let mesh = explicit_mesh();
        let input_sharding = sharding(&mesh, &[Some("x"), Some("y")]);
        let item_sharding = sharding(&mesh, &[Some("y")]);
        let trace = |axis: &'static str| {
            ArrayTracingContext::trace(
                |input| {
                    Ok(batch(
                        |item| Ok(auto_axes(|item| item.sin(), item, Some(&[axis]), item_sharding.clone())?),
                        input,
                        BatchAxis::new(0),
                        BatchAxis::new(0),
                        None,
                    )?)
                },
                matrix_type(Some(input_sharding.clone())),
            )
        };
        let (output_type, program) = trace("y").unwrap();
        assert_eq!(output_type.sharding(), Some(&input_sharding));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[4, 2][sharding={mesh<['x'=2:explicit, 'y'=2:explicit]>, [{'x'}, {'y'}]}] .
                let %1:f32[4, 2][sharding={mesh<['x'=2:explicit, 'y'=2:auto]>, [{'x'}, {}]}] = reshard \
                        [sharding={mesh<['x'=2:explicit, 'y'=2:auto]>, [{'x'}, {}]}] %0
                    %2:f32[4, 2][sharding={mesh<['x'=2:explicit, 'y'=2:auto]>, [{'x'}, {}]}] = sin %1
                    %3:f32[4, 2][sharding={mesh<['x'=2:explicit, 'y'=2:explicit]>, [{'x'}, {'y'}]}] = reshard \
                        [sharding={mesh<['x'=2:explicit, 'y'=2:explicit]>, [{'x'}, {'y'}]}] %2
                in (%3)
            "}
            .trim_end(),
        );
        assert_close(&program.interpret(matrix(Some(input_sharding.clone()))).unwrap(), &SINE);

        // A batch dimension placed along the switched axis loses that placement inside the switch, which leaves it to
        // the compiler, and is placed along `x` again on exit, since the batching level remembers the placement of
        // its batch, as in the JAX oracle `vmap auto_axes(x) over x-placed batch` (`float32[4@x,2@y]`).
        let (output_type, program) = trace("x").unwrap();
        assert_eq!(output_type.sharding(), Some(&input_sharding));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[4, 2][sharding={mesh<['x'=2:explicit, 'y'=2:explicit]>, [{'x'}, {'y'}]}] .
                let %1:f32[4, 2][sharding={mesh<['x'=2:auto, 'y'=2:explicit]>, [{}, {'y'}]}] = reshard \
                        [sharding={mesh<['x'=2:auto, 'y'=2:explicit]>, [{}, {'y'}]}] %0
                    %2:f32[4, 2][sharding={mesh<['x'=2:auto, 'y'=2:explicit]>, [{}, {'y'}]}] = sin %1
                    %3:f32[4, 2][sharding={mesh<['x'=2:explicit, 'y'=2:explicit]>, [{'x'}, {'y'}]}] = reshard \
                        [sharding={mesh<['x'=2:explicit, 'y'=2:explicit]>, [{'x'}, {'y'}]}] %2
                in (%3)
            "}
            .trim_end(),
        );
        assert_close(&program.interpret(matrix(Some(input_sharding.clone()))).unwrap(), &SINE);
    }

    #[test]
    fn test_explicit_axes() {
        // From an all-auto mesh, the body sees `x` as explicit, its input is resharded along `x`, and its output loses
        // its placement along `x` again, as in the JAX oracle `explicit_axes(x)`.
        let mesh = auto_mesh();
        let input_sharding = Sharding::replicated(mesh.clone(), 2);
        let (output_type, program) = ArrayTracingContext::trace(
            |input| {
                Ok(explicit_axes(
                    |input: Tracer<ArrayTracingContext>| input.cos()?.reduce_sum(&[1], None),
                    input,
                    Some(&["x"]),
                    sharding(&mesh, &[Some("x"), None]),
                )?)
            },
            matrix_type(Some(input_sharding.clone())),
        )
        .unwrap();
        assert_eq!(output_type.sharding(), Some(&Sharding::replicated(mesh.clone(), 1)));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[4, 2][sharding={mesh<['x'=2:auto, 'y'=2:auto]>, [{}, {}]}] .
                let %1:f32[4, 2][sharding={mesh<['x'=2:explicit, 'y'=2:auto]>, [{'x'}, {}]}] = reshard \
                        [sharding={mesh<['x'=2:explicit, 'y'=2:auto]>, [{'x'}, {}]}] %0
                    %2:f32[4, 2][sharding={mesh<['x'=2:explicit, 'y'=2:auto]>, [{'x'}, {}]}] = cos %1
                    %3:f32[4][sharding={mesh<['x'=2:explicit, 'y'=2:auto]>, [{'x'}]}] = reduce [kind=sum, axes=[1]] %2
                    %4:f32[4][sharding={mesh<['x'=2:auto, 'y'=2:auto]>, [{}]}] = reshard \
                        [sharding={mesh<['x'=2:auto, 'y'=2:auto]>, [{}]}] %3
                in (%4)
            "}
            .trim_end(),
        );
        let output = explicit_axes(
            |input: Array| input.cos()?.reduce_sum(&[1], None),
            matrix(Some(input_sharding.clone())),
            Some(&["x"]),
            sharding(&mesh, &[Some("x"), None]),
        )
        .unwrap();
        assert_close(&output, &[1.5403022766113281, -1.4061393737792969, -0.36998140811920166, 1.7140724658966064]);
        assert_eq!(program.interpret(matrix(Some(input_sharding))), Ok(output));
    }

    #[test]
    fn test_explicit_axes_rejects_invalid_switches() {
        let mesh = auto_mesh();
        let other_mesh = LogicalMesh::new(vec![MeshAxis::new("z", 4, MeshAxisType::Auto).unwrap()]).unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 2);
        assert_eq!(
            explicit_axes(|inputs: Vec<Array>| Ok(inputs), Vec::<Array>::new(), None, Vec::new()),
            Err(AxisTypesError::MissingMesh { axis_type: MeshAxisType::Explicit }),
        );
        assert_eq!(
            explicit_axes(
                |inputs: (Array, Array)| Ok(inputs.0),
                (matrix(None), matrix(None)),
                None,
                (replicated.clone(), Sharding::replicated(other_mesh.clone(), 2)),
            ),
            Err(AxisTypesError::MeshMismatch {
                axis_type: MeshAxisType::Explicit,
                expected: mesh.clone(),
                actual: other_mesh.clone(),
            }),
        );
        assert_eq!(
            explicit_axes(
                |inputs: Vec<Array>| Ok(inputs),
                vec![matrix(None), matrix(None)],
                None,
                vec![replicated.clone()],
            ),
            Err(AxisTypesError::Parameter(ParameterError::MismatchedParameterStructures {
                left_structure: "[$[0], $[1]]".to_string(),
                right_structure: "[$[0]]".to_string(),
            })),
        );
        let manual_mesh = mesh.with_axis_types(["y"], MeshAxisType::Manual).unwrap();
        assert_eq!(
            explicit_axes(|input: Array| Ok(input), matrix(None), None, Sharding::replicated(manual_mesh, 2)),
            Err(AxisTypesError::ManualAxis { axis_type: MeshAxisType::Explicit, axis_name: "y".to_string() }),
        );
        assert_eq!(
            explicit_axes(
                |input: Array| Ok(input),
                matrix(Some(Sharding::replicated(other_mesh.clone(), 2))),
                Some(&["x"]),
                replicated.clone(),
            ),
            Err(AxisTypesError::InputMeshMismatch {
                axis_type: MeshAxisType::Explicit,
                input_index: 0,
                expected: mesh.clone(),
                actual: other_mesh.clone(),
            }),
        );

        // Outputs must be placed over the mesh of the switch, and they cannot carry reduction state along an axis that
        // is auto outside of the switch.
        let captured = matrix(Some(Sharding::replicated(other_mesh.clone(), 2)));
        assert_eq!(
            explicit_axes(|_: Array| Ok(captured.clone()), matrix(None), Some(&["x"]), replicated.clone()),
            Err(AxisTypesError::OutputMeshMismatch {
                axis_type: MeshAxisType::Explicit,
                output_index: 0,
                expected: mesh.clone(),
                actual: other_mesh,
            }),
        );
        let switched_mesh = mesh.with_axis_types(["x"], MeshAxisType::Explicit).unwrap();
        let unreduced = matrix(Some(Sharding::replicated(switched_mesh, 2).with_unreduced_axes(["x"]).unwrap()));
        assert_eq!(
            explicit_axes(
                |input: Array| Ok((input, unreduced.clone())),
                matrix(None),
                Some(&["x"]),
                replicated.clone()
            ),
            Err(AxisTypesError::OutputReductionState { output_index: 1, axis_name: "x".to_string() }),
        );

        // The reshards validate the input shardings, which cannot name an axis that stays auto.
        assert_eq!(
            explicit_axes(|input: Array| Ok(input), matrix(None), Some(&["x"]), sharding(&mesh, &[Some("y"), None])),
            Err(AxisTypesError::Program(ProgramError::Type(TypeError::invalid(RESHARD_AUTO_TARGET_ERROR)))),
        );
    }

    #[test]
    fn test_explicit_axes_differentiation() {
        // Reverse mode differentiates through the reshards and agrees with the JAX oracle `grad explicit_axes`.
        let mesh = auto_mesh();
        let input_sharding = Sharding::replicated(mesh.clone(), 2);
        let gradient = differentiate_at(matrix(Some(input_sharding.clone())))
            .gradient(|input| {
                Ok(explicit_axes(
                    |input| input.cos()?.mul(&input)?.reduce_sum(&[0, 1], None),
                    input,
                    Some(&["x"]),
                    sharding(&mesh, &[Some("x"), None]),
                )?)
            })
            .unwrap();
        assert_eq!(gradient.r#type().sharding(), Some(&input_sharding));
        assert_close(
            &gradient,
            &[
                1.0,
                -0.30116868019104004,
                -2.234741687774658,
                -1.4133524894714355,
                2.3735663890838623,
                5.078283786773682,
                2.6366631984710693,
                -3.845003843307495,
            ],
        );

        // The JAX oracle places its input along both auto axes, which JAX types never record. A Ryft type may record
        // such a placement (e.g., that of a concrete input). The transposition of the entry reshard cannot target auto
        // axes, so it reshards only the explicit part and restores the auto placement with a placement-only broadcast,
        // and the gradient has the input's cotangent type with the same values.
        let placed_input_type = matrix_type(Some(sharding(&mesh, &[Some("x"), Some("y")])));
        let gradient = differentiate_at(matrix(placed_input_type.sharding().cloned()))
            .gradient(|input| {
                Ok(explicit_axes(
                    |input| input.cos()?.mul(&input)?.reduce_sum(&[0, 1], None),
                    input,
                    Some(&["x"]),
                    sharding(&mesh, &[Some("x"), None]),
                )?)
            })
            .unwrap();
        assert_eq!(gradient.r#type().into_owned(), placed_input_type.cotangent().unwrap());
        assert_eq!(gradient.r#type().sharding(), placed_input_type.sharding());
        assert_close(
            &gradient,
            &[
                1.0,
                -0.30116868019104004,
                -2.234741687774658,
                -1.4133524894714355,
                2.3735663890838623,
                5.078283786773682,
                2.6366631984710693,
                -3.845003843307495,
            ],
        );
    }

    #[test]
    fn test_explicit_axes_inside_auto_axes() {
        // An `explicit_axes` nested in an `auto_axes` over every axis makes `x` explicit again for its body, whose
        // input is placed along `x` (JAX's `float32[4@x,2]`) and whose output is replicated over the all-auto mesh of
        // the outer body (JAX's `float32[4,2]`).
        let mesh = explicit_mesh();
        let input_sharding = sharding(&mesh, &[Some("x"), Some("y")]);
        let (_, program) = ArrayTracingContext::trace(
            |input| {
                Ok(auto_axes(
                    |input: Tracer<ArrayTracingContext>| {
                        let doubled = explicit_axes(
                            |input: Tracer<ArrayTracingContext>| input.add(&input),
                            input.clone(),
                            Some(&["x"]),
                            sharding(&auto_mesh(), &[Some("x"), None]),
                        )?;
                        doubled.add(&input)
                    },
                    input,
                    None,
                    input_sharding.clone(),
                )?)
            },
            matrix_type(Some(input_sharding.clone())),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[4, 2][sharding={mesh<['x'=2:explicit, 'y'=2:explicit]>, [{'x'}, {'y'}]}] .
                let %1:f32[4, 2][sharding={mesh<['x'=2:auto, 'y'=2:auto]>, [{}, {}]}] = reshard \
                        [sharding={mesh<['x'=2:auto, 'y'=2:auto]>, [{}, {}]}] %0
                    %2:f32[4, 2][sharding={mesh<['x'=2:explicit, 'y'=2:auto]>, [{'x'}, {}]}] = reshard \
                        [sharding={mesh<['x'=2:explicit, 'y'=2:auto]>, [{'x'}, {}]}] %1
                    %3:f32[4, 2][sharding={mesh<['x'=2:explicit, 'y'=2:auto]>, [{'x'}, {}]}] = add %2 %2
                    %4:f32[4, 2][sharding={mesh<['x'=2:auto, 'y'=2:auto]>, [{}, {}]}] = reshard \
                        [sharding={mesh<['x'=2:auto, 'y'=2:auto]>, [{}, {}]}] %3
                    %5:f32[4, 2][sharding={mesh<['x'=2:auto, 'y'=2:auto]>, [{}, {}]}] = add %4 %1
                    %6:f32[4, 2][sharding={mesh<['x'=2:explicit, 'y'=2:explicit]>, [{'x'}, {'y'}]}] = reshard \
                        [sharding={mesh<['x'=2:explicit, 'y'=2:explicit]>, [{'x'}, {'y'}]}] %5
                in (%6)
            "}
            .trim_end(),
        );
        assert_eq!(
            program.interpret(matrix(Some(input_sharding))).map(|output| output.to_f64s()),
            Ok(vec![0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0]),
        );
    }
}
