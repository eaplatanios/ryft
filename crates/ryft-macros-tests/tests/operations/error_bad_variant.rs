use std::marker::PhantomData;

struct DataType;
struct TypeError;

trait Value {
    type Type;
}

struct RegionInterface<T> {
    marker: PhantomData<T>,
}

struct OutputRegionProvenance {
    region_index: usize,
    output_index: usize,
}

struct RegionSlot;

trait Operation {
    type Type;

    fn name(&self) -> &'static str;

    fn region_slots(&self) -> &'static [RegionSlot] {
        &[]
    }

    fn infer_output_types(
        &self,
        input_types: &[Self::Type],
        region_interfaces: &[RegionInterface<Self::Type>],
    ) -> Result<Vec<Self::Type>, TypeError>;

    fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
        let _ = output_index;
        Vec::new()
    }
}

#[derive(ryft::Operation)]
enum BadOperation<V: Value<Type = DataType>> {
    Add,
    Marker(PhantomData<V>),
}

fn main() {}
