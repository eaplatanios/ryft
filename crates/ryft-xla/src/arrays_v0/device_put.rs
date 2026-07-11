use crate::experimental::domains::XlaDomain;
use crate::{Array, FromPjrt};

use ryft_core::{ArrayType, Shape, Size};

use super::*;

/// Leaf types accepted by the higher-level [`device_put()`] API.
///
/// A [`DevicePutLeaf`] consumes one input leaf and materializes one runtime [`Array`] leaf. `ryft`
/// currently provides implementations for runtime [`Array`] leaves, primitive scalar host values,
/// and owned `ndarray::Array`s when the `ndarray` feature is enabled.
pub trait DevicePutLeaf<'c>: Parameter {
    /// Converts `self` into one runtime [`Array`] using the provided leafwise placement options.
    ///
    /// # Parameters
    ///
    ///   - `engine`: [`XlaDomain`] wrapping the PJRT client used to materialize any needed
    ///     destination buffers and to cache any compiled reshard executables.
    ///   - `device`: Destination placement for this leaf, if one was specified.
    ///   - `src`: Source placement for this leaf, if one was specified.
    ///   - `donate`: Best-effort donation flag for this leaf.
    ///   - `may_alias`: Best-effort aliasing hint for this leaf.
    fn device_put_leaf(
        self,
        engine: &XlaDomain<'c>,
        device: Option<DevicePutTarget>,
        src: Option<DevicePutTarget>,
        donate: bool,
        may_alias: Option<bool>,
    ) -> Result<Array<'c>, ArrayError>;
}

impl<'c, T: DenseHostDevicePutLeaf + Parameter> DevicePutLeaf<'c> for T {
    fn device_put_leaf(
        self,
        engine: &XlaDomain<'c>,
        device: Option<DevicePutTarget>,
        _src: Option<DevicePutTarget>,
        _donate: bool,
        _may_alias: Option<bool>,
    ) -> Result<Array<'c>, ArrayError> {
        let client = engine.client()?;
        let (shape, element_type, bytes) = self.into_dense_host_array();
        let (mesh, sharding) = match device {
            Some(device) => device.resolve(shape.len())?,
            None => {
                let device =
                    client.addressable_devices()?.into_iter().next().ok_or(ArrayError::MissingDefaultDevice)?;
                DevicePutTarget::device(Device::from_pjrt(device)?).resolve(shape.len())?
            }
        };
        let r#type = ArrayType::new(element_type, Shape::new(shape.iter().copied().map(Size::Static).collect()))
            .with_sharding(sharding)?;
        Array::from_host_buffer(client, r#type, mesh, bytes.as_slice())
    }
}

impl<'c> DevicePutLeaf<'c> for Array<'c> {
    fn device_put_leaf(
        self,
        engine: &XlaDomain<'c>,
        device: Option<DevicePutTarget>,
        src: Option<DevicePutTarget>,
        _donate: bool,
        may_alias: Option<bool>,
    ) -> Result<Array<'c>, ArrayError> {
        let current_mesh = self.mesh();
        let current_sharding = self.sharding().clone();
        if let Some(src) = src {
            let (expected_mesh, expected_sharding) = src.resolve(current_sharding.rank())?;
            if expected_mesh != current_mesh || expected_sharding != current_sharding {
                return Err(ArrayError::SourcePlacementMismatch {
                    expected_mesh,
                    expected_sharding,
                    actual_mesh: current_mesh.clone(),
                    actual_sharding: current_sharding.clone(),
                });
            }
        }

        let (target_mesh, target_sharding) = match device {
            Some(device) => device.resolve(self.sharding().rank())?,
            None => (current_mesh.clone(), current_sharding.clone()),
        };
        if target_mesh == current_mesh && target_sharding == current_sharding && may_alias != Some(false) {
            Ok(self)
        } else {
            self.to_placement(engine, DevicePutTarget::placement(target_mesh, target_sharding)?)
        }
    }
}

/// Higher-level `ryft` analogue of JAX's
/// [`jax.device_put`](https://docs.jax.dev/en/latest/_autosummary/jax.device_put.html).
///
/// The input `x` may be one supported leaf or a `Parameterized` tree of supported leaves. Any
/// provided `device`, `src`, `donate`, and `may_alias` fields follow tree-prefix broadcasting
/// semantics over `x`.
///
/// Host leaves are committed to the default local device when `options.device` is absent. Existing
/// [`Array`] leaves preserve their current placement when `options.device` is absent.
pub fn device_put<
    'c,
    P: DevicePutLeaf<'c>,
    Input: Parameterized<
            P,
            Family: ParameterizedFamily<Array<'c>>
                        + ParameterizedFamily<DevicePutTarget>
                        + ParameterizedFamily<bool>
                        + ParameterizedFamily<Option<bool>>,
        >,
    DeviceTarget: Parameterized<DevicePutTarget>,
    SourceTarget: Parameterized<DevicePutTarget>,
    Donate: Parameterized<bool>,
    MayAlias: Parameterized<Option<bool>>,
>(
    engine: &XlaDomain<'c>,
    x: Input,
    options: DevicePutOptions<DeviceTarget, SourceTarget, Donate, MayAlias>,
) -> Result<<Input as Parameterized<P>>::To<Array<'c>>, ArrayError> {
    let structure = x.parameter_structure();
    let leaf_count = structure.parameter_count();
    let (device, src, donate, may_alias) = options.into_parts();

    let device_values = match device {
        Some(device) => Input::To::<DevicePutTarget>::from_broadcasted_named_parameters(
            structure.clone(),
            device.into_named_parameters(),
        )?
        .into_parameters()
        .map(Some)
        .collect::<Vec<_>>(),
        None => vec![None; leaf_count],
    };
    let src_values = match src {
        Some(src) => Input::To::<DevicePutTarget>::from_broadcasted_named_parameters(
            structure.clone(),
            src.into_named_parameters(),
        )?
        .into_parameters()
        .map(Some)
        .collect::<Vec<_>>(),
        None => vec![None; leaf_count],
    };
    let donate_values = match donate {
        Some(donate) => {
            Input::To::<bool>::from_broadcasted_named_parameters(structure.clone(), donate.into_named_parameters())?
                .into_parameters()
                .collect::<Vec<_>>()
        }
        None => vec![false; leaf_count],
    };
    let may_alias_values = match may_alias {
        Some(may_alias) => Input::To::<Option<bool>>::from_broadcasted_named_parameters(
            structure.clone(),
            may_alias.into_named_parameters(),
        )?
        .into_parameters()
        .collect::<Vec<_>>(),
        None => vec![None; leaf_count],
    };

    let mut output_parameters = Vec::with_capacity(leaf_count);
    let mut device_values = device_values.into_iter();
    let mut src_values = src_values.into_iter();
    let mut donate_values = donate_values.into_iter();
    let mut may_alias_values = may_alias_values.into_iter();
    for parameter in x.into_parameters() {
        output_parameters.push(
            parameter.device_put_leaf(
                engine,
                device_values
                    .next()
                    .expect("device tree-prefix broadcasting should produce one placement per input leaf"),
                src_values.next().expect("src tree-prefix broadcasting should produce one placement per input leaf"),
                donate_values
                    .next()
                    .expect("donate tree-prefix broadcasting should produce one flag per input leaf"),
                may_alias_values
                    .next()
                    .expect("may_alias tree-prefix broadcasting should produce one flag per input leaf"),
            )?,
        );
    }
    Input::To::<Array<'c>>::from_parameters(structure, output_parameters).map_err(Into::into)
}
