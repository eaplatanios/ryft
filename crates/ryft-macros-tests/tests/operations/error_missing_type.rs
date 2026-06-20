trait Operation<T> {}

#[derive(ryft::Operation)]
enum BadOperation {
    Add(AddOperation),
}

struct AddOperation;

fn main() {}
