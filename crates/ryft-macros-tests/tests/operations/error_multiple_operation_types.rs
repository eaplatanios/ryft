use ryft::{ArrayType, DataType, Operation, RegionInterface, TypeError};

#[derive(Clone)]
struct BadOperation;

impl Operation for BadOperation {
    type Type = DataType;

    fn name(&self) -> &'static str {
        "bad"
    }

    fn infer_output_types(
        &self,
        input_types: &[DataType],
        _region_interfaces: &[RegionInterface<DataType>],
    ) -> Result<Vec<DataType>, TypeError> {
        Ok(input_types.to_vec())
    }
}

impl Operation for BadOperation {
    type Type = ArrayType;

    fn name(&self) -> &'static str {
        "bad"
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        Ok(input_types.to_vec())
    }
}

fn main() {}
