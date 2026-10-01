struct ArrayType;

trait Value {
    type Type;
}

struct DotOperation;

#[derive(ryft::Operation)]
enum BadOperation<V: Value<Type = ArrayType>> {
    Dot(DotOperation),
    BoxedDot(Box<DotOperation>),
    Extension(V),
}

fn main() {}
