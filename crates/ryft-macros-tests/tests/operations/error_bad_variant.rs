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

trait Operation<T> {
    fn name(&self) -> &'static str;

    fn infer_output_types(
        &self,
        input_types: &[T],
        region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<T>, TypeError>;

    fn region_names(&self) -> &'static [&'static str] {
        &[]
    }

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
