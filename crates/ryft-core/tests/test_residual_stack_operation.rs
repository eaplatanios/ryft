use std::fmt::{Display, Formatter};

use pretty_assertions::assert_eq;

use ryft_core::{NoIdentity, Operation, Parameter, RegionInterface, ResidualStackOperation, Type, TypeError};

#[test]
fn test_external_residual_stack_operation() {
    // An independent type universe selects its own array encoding without implementing a runtime value family.
    #[derive(Clone, Debug, PartialEq)]
    enum ExternalType {
        Metadata,
        Array,
    }

    impl Display for ExternalType {
        fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::Metadata => formatter.write_str("metadata"),
                Self::Array => formatter.write_str("i64[]"),
            }
        }
    }

    impl Parameter for ExternalType {}

    impl Type for ExternalType {
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

    #[derive(Clone)]
    enum ExternalOperation {
        Save,
        Restore,
    }

    impl Operation for ExternalOperation {
        type Type = ExternalType;

        fn name(&self) -> &'static str {
            match self {
                Self::Save => "save_metadata",
                Self::Restore => "restore_metadata",
            }
        }

        fn infer_output_types(
            &self,
            input_types: &[ExternalType],
            region_interfaces: &[RegionInterface<ExternalType>],
        ) -> Result<Vec<ExternalType>, TypeError> {
            match (self, input_types, region_interfaces) {
                (Self::Save, [ExternalType::Metadata], []) => Ok(vec![ExternalType::Array]),
                (Self::Restore, [ExternalType::Array], []) => Ok(vec![ExternalType::Metadata]),
                _ => Err(TypeError::invalid("invalid external residual conversion")),
            }
        }
    }

    impl ResidualStackOperation<ExternalType> for ExternalOperation {
        fn residual_to_storage(residual_type: &ExternalType) -> Result<Option<Self>, TypeError> {
            Ok(matches!(residual_type, ExternalType::Metadata).then_some(Self::Save))
        }

        fn residual_from_storage(residual_type: &ExternalType) -> Result<Option<Self>, TypeError> {
            Ok(matches!(residual_type, ExternalType::Metadata).then_some(Self::Restore))
        }

        fn residual_from_carrier(
            _residual_type: &ExternalType,
            _carrier_type: &ExternalType,
        ) -> Result<Option<Self>, TypeError> {
            Ok(None)
        }
    }

    let storage_type = ExternalType::Array;
    assert_eq!(ExternalOperation::residual_storage_type(&ExternalType::Metadata), Ok(storage_type.clone()));
    assert_eq!(ExternalOperation::residual_storage_type(&storage_type), Ok(storage_type.clone()));
    let restore = ExternalOperation::residual_from_storage(&ExternalType::Metadata).unwrap().unwrap();
    assert_eq!(restore.infer_output_types(&[storage_type], &[]), Ok(vec![ExternalType::Metadata]));
}
