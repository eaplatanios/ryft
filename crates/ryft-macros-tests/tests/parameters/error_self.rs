use ryft::{Parameter, Parameterized};

// TODO(eaplatanios): Some of these cases should not result in errors. Figure out if we can improve robustness.

pub trait Trait {
    type Assoc;
}

#[derive(Parameterized)]
pub struct Generics<P: Parameter, T: Trait<Assoc = Self>>
where
    Self: Trait<Assoc = Self>,
    <Self as Trait>::Assoc: Sized,
{
    _f: T,
    _p: P,
}

impl<P: Parameter, T: Trait<Assoc = Self>> Trait for Generics<P, T> {
    type Assoc = Self;
}

#[derive(Parameterized)]
pub struct Struct<P: Parameter> {
    _p0: P,
    _f1: Box<Self>,
    _f2: Box<<Self as Trait>::Assoc>,
    _f4: [(); Self::ASSOC],
    _f5: [(); Self::assoc()],
}

impl<P: Parameter> Struct<P> {
    const ASSOC: usize = 1;
    const fn assoc() -> usize {
        0
    }
}

impl<P: Parameter> Trait for Struct<P> {
    type Assoc = Self;
}

#[derive(Parameterized)]
pub struct Tuple<P: Parameter>(Box<Self>, Box<<Self as Trait>::Assoc>, [(); Self::ASSOC], [(); Self::assoc()]);

impl<P: Parameter> Tuple<P> {
    const ASSOC: usize = 1;
    const fn assoc() -> usize {
        0
    }
}

impl<P: Parameter> Trait for Tuple<P> {
    type Assoc = Self;
}

#[derive(Parameterized)]
pub enum Enum<P: Parameter> {
    Struct {
        _p0: P,
        _f1: Box<Self>,
        _f2: Box<<Self as Trait>::Assoc>,
        _f4: [(); Self::ASSOC],
        _f5: [(); Self::assoc()],
    },
    Tuple(Box<Self>, Box<<Self as Trait>::Assoc>, [(); Self::ASSOC], [(); Self::assoc()]),
}

impl<P: Parameter> Enum<P> {
    const ASSOC: usize = 1;
    const fn assoc() -> usize {
        0
    }
}

impl<P: Parameter> Trait for Enum<P> {
    type Assoc = Self;
}

fn main() {}
