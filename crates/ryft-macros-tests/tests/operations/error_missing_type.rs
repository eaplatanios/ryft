trait Operation {
    type Type;
}

#[derive(ryft::Operation)]
enum BadOperation {
    Add(AddOperation),
}

struct AddOperation;

fn main() {}
