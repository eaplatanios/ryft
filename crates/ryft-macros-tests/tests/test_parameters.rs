use std::marker::PhantomData;
use std::ops::{Add, Mul};

use ryft::{Parameter, Parameterized, Placeholder};

/// Helper for asserting that a [`Parameterized`] type has a specific [`Parameterized::ParamStructure`] type.
fn assert_param_structure_type<P: Parameter, T: Parameterized<P, ParamStructure = S>, S>() {}

#[test]
fn test_simple_struct() {
    #[derive(Parameterized, Debug, Clone, PartialEq, Eq)]
    struct Struct<P: Parameter> {
        p_0: P,
        p_1: P,
        np_0: (usize, usize),
        np_1: u64,
    }

    let np_0 = (4usize, 2usize);
    let np_1 = 42u64;
    let mut value = Struct { p_0: 4usize, p_1: 2usize, np_0, np_1 };
    let structure = Struct { p_0: Placeholder, p_1: Placeholder, np_0, np_1 };
    let insufficient_params_error = Err(ryft::Error::InsufficientParams { expected_count: 2 });

    assert_param_structure_type::<usize, Struct<usize>, Struct<Placeholder>>();
    assert_eq!(value.param_count(), 2);
    assert_eq!(value.param_structure(), structure);
    assert_eq!(value.params().collect::<Vec<_>>(), vec![&4usize, &2usize]);
    assert_eq!(value.params_mut().collect::<Vec<_>>(), vec![&mut 4usize, &mut 2usize]);
    assert_eq!(value.clone().into_params().collect::<Vec<_>>(), vec![4usize, 2usize]);
    assert_eq!(Struct::from_params(structure.clone(), Vec::<usize>::new()), insufficient_params_error);
    assert_eq!(
        Struct::from_params(structure.clone(), vec![4i64, 2i64]),
        Ok(Struct { p_0: 4i64, p_1: 2i64, np_0, np_1 })
    );
    assert_eq!(Struct::from_params(structure, [0usize; 10]), Err(ryft::Error::UnusedParams));

    value.params_mut().for_each(|param| *param *= 2);
    assert_eq!(value, Struct { p_0: 8usize, p_1: 4usize, np_0, np_1 })
}

#[test]
fn test_tuple_struct() {
    #[derive(Parameterized, Debug, Clone, PartialEq, Eq)]
    #[ryft(crate = "ryft")]
    struct TupleStruct<P: Parameter>(P, P, (usize, usize), u64);

    let np_0 = (4usize, 2usize);
    let np_1 = 42u64;
    let mut value = TupleStruct(4usize, 2usize, np_0, np_1);
    let structure = TupleStruct(Placeholder, Placeholder, np_0, np_1);
    let insufficient_params_error = Err(ryft::Error::InsufficientParams { expected_count: 2 });

    assert_param_structure_type::<usize, TupleStruct<usize>, TupleStruct<Placeholder>>();
    assert_eq!(value.param_count(), 2);
    assert_eq!(value.param_structure(), structure);
    assert_eq!(value.params().collect::<Vec<_>>(), vec![&4usize, &2usize]);
    assert_eq!(value.params_mut().collect::<Vec<_>>(), vec![&mut 4usize, &mut 2usize]);
    assert_eq!(value.clone().into_params().collect::<Vec<_>>(), vec![4usize, 2usize]);
    assert_eq!(TupleStruct::from_params(structure.clone(), Vec::<usize>::new()), insufficient_params_error);
    assert_eq!(TupleStruct::from_params(structure.clone(), vec![4i64, 2i64]), Ok(TupleStruct(4i64, 2i64, np_0, np_1)));
    assert_eq!(TupleStruct::from_params(structure, [0usize; 10]), Err(ryft::Error::UnusedParams));
}

#[test]
fn test_struct_with_nested_tuples() {
    #[derive(Parameterized, Debug, Clone, PartialEq, Eq)]
    struct StructWithNestedTuples<P: Parameter> {
        p_0: P,
        p_1: (usize, (i32, P, i64, P)),
        np_0: (usize, usize),
        np_1: u64,
    }

    let np_0 = (4usize, 2usize);
    let np_1 = 42u64;
    let mut value = StructWithNestedTuples { p_0: 4usize, p_1: (0usize, (-1i32, 2usize, -42i64, 0usize)), np_0, np_1 };
    let structure = StructWithNestedTuples {
        p_0: Placeholder,
        p_1: (0usize, (-1i32, Placeholder, -42i64, Placeholder)),
        np_0,
        np_1,
    };
    let insufficient_params_error = Err(ryft::Error::InsufficientParams { expected_count: 3 });

    assert_eq!(value.param_count(), 3);
    assert_eq!(value.param_structure(), structure);
    assert_eq!(value.params().collect::<Vec<_>>(), vec![&4usize, &2usize, &0usize]);
    assert_eq!(value.params_mut().collect::<Vec<_>>(), vec![&mut 4usize, &mut 2usize, &mut 0usize]);
    assert_eq!(value.clone().into_params().collect::<Vec<_>>(), vec![4usize, 2usize, 0usize]);
    assert_eq!(StructWithNestedTuples::from_params(structure.clone(), Vec::<usize>::new()), insufficient_params_error);
    assert_eq!(
        StructWithNestedTuples::from_params(structure.clone(), vec![4i64, 2i64, 0i64]),
        Ok(StructWithNestedTuples { p_0: 4i64, p_1: (0usize, (-1i32, 2i64, -42i64, 0i64)), np_0, np_1 })
    );
    assert_eq!(StructWithNestedTuples::from_params(structure, [0usize; 10]), Err(ryft::Error::UnusedParams));
}

#[test]
fn test_struct_with_nested_struct() {
    #[derive(Parameterized, Debug, Clone, PartialEq, Eq)]
    struct Struct<P: Parameter + Clone> {
        p_0: P,
        p_1: P,
        np_0: (usize, usize),
        np_1: u64,
    }

    #[derive(Parameterized, Debug, Clone, PartialEq, Eq)]
    struct StructWithNestedStruct<P: Parameter>
    where
        P: Clone,
    {
        p_0: P,
        p_1: (usize, (i32, P, i64, Struct<P>)),
        np_0: (usize, usize),
        np_1: u64,
    }

    let np_0 = (4usize, 2usize);
    let np_1 = 42u64;
    let mut value = StructWithNestedStruct {
        p_0: 4usize,
        p_1: (0usize, (-1i32, 2usize, -42i64, Struct { p_0: 4usize, p_1: 2usize, np_0, np_1 })),
        np_0,
        np_1,
    };
    let structure = StructWithNestedStruct {
        p_0: Placeholder,
        p_1: (0usize, (-1i32, Placeholder, -42i64, Struct { p_0: Placeholder, p_1: Placeholder, np_0, np_1 })),
        np_0,
        np_1,
    };
    let insufficient_params_error = Err(ryft::Error::InsufficientParams { expected_count: 4 });

    assert_eq!(value.param_count(), 4);
    assert_eq!(value.param_structure(), structure);
    assert_eq!(value.params().collect::<Vec<_>>(), vec![&4usize, &2usize, &4usize, &2usize]);
    assert_eq!(value.params_mut().collect::<Vec<_>>(), vec![&mut 4usize, &mut 2usize, &mut 4usize, &mut 2usize]);
    assert_eq!(value.clone().into_params().collect::<Vec<_>>(), vec![4usize, 2usize, 4usize, 2usize]);
    assert_eq!(StructWithNestedStruct::from_params(structure.clone(), Vec::<usize>::new()), insufficient_params_error);
    assert_eq!(
        StructWithNestedStruct::from_params(structure.clone(), vec![-4i64, -2i64, 4i64, 2i64]),
        Ok(StructWithNestedStruct {
            p_0: -4i64,
            p_1: (0usize, (-1i32, -2i64, -42i64, Struct { p_0: 4i64, p_1: 2i64, np_0, np_1 })),
            np_0,
            np_1,
        })
    );
    assert_eq!(StructWithNestedStruct::from_params(structure, [0usize; 10]), Err(ryft::Error::UnusedParams));
}

#[test]
fn test_empty_struct() {
    #[derive(Parameterized, Debug, Clone, PartialEq, Eq)]
    struct EmptyStruct<P: Parameter> {
        _p: PhantomData<P>,
    }

    let mut value = EmptyStruct { _p: PhantomData::<usize> };
    let structure = EmptyStruct { _p: PhantomData::<Placeholder> };

    assert_eq!(value.param_count(), 0);
    assert_eq!(value.param_structure(), structure);
    assert_eq!(value.params().collect::<Vec<_>>(), Vec::<&usize>::new());
    assert_eq!(value.params_mut().collect::<Vec<_>>(), Vec::<&mut usize>::new());
    assert_eq!(value.clone().into_params().collect::<Vec<_>>(), Vec::<usize>::new());
    assert_eq!(
        EmptyStruct::from_params(structure.clone(), Vec::<i64>::new()),
        Ok(EmptyStruct { _p: PhantomData::<i64> })
    );
    assert_eq!(EmptyStruct::from_params(structure, vec![4i64, 2i64]), Err(ryft::Error::UnusedParams));
}

#[test]
fn test_non_empty_struct_with_no_parameters() {
    #[derive(Parameterized, Debug, Clone, PartialEq, Eq)]
    struct NonEmptyStruct<P: Parameter> {
        np_0: (usize, usize),
        np_1: u64,
        _p: PhantomData<P>,
    }

    let np_0 = (4usize, 2usize);
    let np_1 = 42u64;
    let mut value = NonEmptyStruct { np_0, np_1, _p: PhantomData::<usize> };
    let structure = NonEmptyStruct { np_0, np_1, _p: PhantomData::<Placeholder> };

    assert_eq!(value.param_count(), 0);
    assert_eq!(value.param_structure(), structure);
    assert_eq!(value.params().collect::<Vec<_>>(), Vec::<&usize>::new());
    assert_eq!(value.params_mut().collect::<Vec<_>>(), Vec::<&mut usize>::new());
    assert_eq!(value.clone().into_params().collect::<Vec<_>>(), Vec::<usize>::new());
    assert_eq!(
        NonEmptyStruct::from_params(structure.clone(), Vec::<i64>::new()),
        Ok(NonEmptyStruct { np_0, np_1, _p: PhantomData::<i64> })
    );
    assert_eq!(NonEmptyStruct::from_params(structure, vec![4usize, 2usize]), Err(ryft::Error::UnusedParams));
}

#[test]
fn test_struct_with_lifetime() {
    #[derive(Parameterized, Debug, Clone, PartialEq, Eq)]
    struct Struct1<'p, V: Parameter> {
        x: V,
        y: (&'p str, usize),
    }

    let mut value = Struct1 { x: 4usize, y: ("hey there", 42usize) };
    let structure = Struct1 { x: Placeholder, y: ("hey there", 42usize) };
    let insufficient_params_error = Err(ryft::Error::InsufficientParams { expected_count: 1 });

    assert_eq!(value.param_count(), 1);
    assert_eq!(value.param_structure(), structure);
    assert_eq!(value.params().collect::<Vec<_>>(), vec![&4usize]);
    assert_eq!(value.params_mut().collect::<Vec<_>>(), vec![&mut 4usize]);
    assert_eq!(value.clone().into_params().collect::<Vec<_>>(), vec![4usize]);
    assert_eq!(Struct1::from_params(structure.clone(), Vec::<usize>::new()), insufficient_params_error);
    assert_eq!(Struct1::from_params(structure.clone(), vec![4i64]), Ok(Struct1 { x: 4i64, y: ("hey there", 42usize) }));
    assert_eq!(Struct1::from_params(structure, [0usize; 10]), Err(ryft::Error::UnusedParams));

    #[derive(Parameterized, Debug, Clone, PartialEq, Eq)]
    struct Struct2<'p, V: Parameter> {
        x: V,
        y: (&'p str, V),
    }

    let mut value = Struct2 { x: 4usize, y: ("hey there", 42usize) };
    let structure = Struct2 { x: Placeholder, y: ("hey there", Placeholder) };
    let insufficient_params_error = Err(ryft::Error::InsufficientParams { expected_count: 2 });

    assert_eq!(value.param_count(), 2);
    assert_eq!(value.param_structure(), structure);
    assert_eq!(value.params().collect::<Vec<_>>(), vec![&4usize, &42usize]);
    assert_eq!(value.params_mut().collect::<Vec<_>>(), vec![&mut 4usize, &mut 42usize]);
    assert_eq!(value.clone().into_params().collect::<Vec<_>>(), vec![4usize, 42usize]);
    assert_eq!(Struct2::from_params(structure.clone(), Vec::<usize>::new()), insufficient_params_error);
    assert_eq!(
        Struct2::from_params(structure.clone(), vec![4i64, 2i64]),
        Ok(Struct2 { x: 4i64, y: ("hey there", 2i64) })
    );
    assert_eq!(Struct2::from_params(structure, [0usize; 10]), Err(ryft::Error::UnusedParams));
}

#[test]
fn test_simple_enum() {
    #[derive(Parameterized, Debug, Clone, PartialEq, Eq)]
    enum Enum<P: Parameter> {
        Unit,
        Empty(),
        Parameter0(P),
        Parameter1 { field_0: (usize, P, usize), field_1: usize },
        NonParameter(usize, usize),
    }

    let structure = Enum::<Placeholder>::Unit;
    assert_eq!(structure.param_count(), 0);
    assert_eq!(structure.param_structure(), structure);
    assert_eq!(structure.params().collect::<Vec<_>>(), Vec::<&Placeholder>::new());
    assert_eq!(Enum::from_params(structure.clone(), Vec::new()), Ok(structure));

    let structure = Enum::<Placeholder>::Empty();
    assert_eq!(structure.param_count(), 0);
    assert_eq!(structure.param_structure(), structure);
    assert_eq!(structure.params().collect::<Vec<_>>(), Vec::<&Placeholder>::new());
    assert_eq!(Enum::from_params(structure.clone(), Vec::new()), Ok(structure));

    let structure = Enum::Parameter0(Placeholder);
    assert_eq!(structure.param_count(), 1);
    assert_eq!(structure.param_structure(), structure);
    assert_eq!(structure.params().collect::<Vec<_>>(), vec![&Placeholder]);
    assert_eq!(Enum::from_params(structure.clone(), vec![42usize]), Ok(Enum::Parameter0(42usize)));

    let mut value = Enum::Parameter1 { field_0: (0usize, -42i64, 4usize), field_1: 2usize };
    let structure = Enum::Parameter1 { field_0: (0usize, Placeholder, 4usize), field_1: 2usize };
    let insufficient_params_error = Err(ryft::Error::InsufficientParams { expected_count: 1 });
    let unused_params_error = Err(ryft::Error::UnusedParams);

    assert_param_structure_type::<usize, Enum<usize>, Enum<Placeholder>>();
    assert_eq!(value.param_count(), 1);
    assert_eq!(value.param_structure(), structure);
    assert_eq!(value.params().collect::<Vec<_>>(), vec![&-42i64]);
    assert_eq!(value.params_mut().collect::<Vec<_>>(), vec![&mut -42i64]);
    assert_eq!(value.clone().into_params().collect::<Vec<_>>(), vec![-42i64]);
    assert_eq!(Enum::from_params(structure.clone(), Vec::<usize>::new()), insufficient_params_error);
    assert_eq!(
        Enum::from_params(structure.clone(), vec![42usize]),
        Ok(Enum::Parameter1 { field_0: (0usize, 42usize, 4usize), field_1: 2usize })
    );
    assert_eq!(Enum::from_params(structure, [0usize; 10]), unused_params_error);

    let structure = Enum::<Placeholder>::NonParameter(4usize, 2usize);
    assert_eq!(structure.param_count(), 0);
    assert_eq!(structure.param_structure(), structure);
    assert_eq!(structure.params().collect::<Vec<_>>(), Vec::<&Placeholder>::new());
    assert_eq!(Enum::from_params(structure.clone(), Vec::new()), Ok(structure));
}

#[test]
fn test_empty_enum() {
    #[derive(Parameterized, Debug, Clone, PartialEq, Eq)]
    enum EmptyEnum<P: Parameter> {
        Phantom(PhantomData<P>),
    }

    let mut value = EmptyEnum::Phantom(PhantomData::<usize>);
    let structure = EmptyEnum::Phantom(PhantomData::<Placeholder>);

    assert_eq!(value.param_count(), 0);
    assert_eq!(value.param_structure(), structure);
    assert_eq!(value.params().collect::<Vec<_>>(), Vec::<&usize>::new());
    assert_eq!(value.params_mut().collect::<Vec<_>>(), Vec::<&mut usize>::new());
    assert_eq!(value.clone().into_params().collect::<Vec<_>>(), Vec::<usize>::new());
    assert_eq!(
        EmptyEnum::from_params(structure.clone(), Vec::<i64>::new()),
        Ok(EmptyEnum::Phantom(PhantomData::<i64>))
    );
    assert_eq!(EmptyEnum::from_params(structure, vec![0usize, 1usize]), Err(ryft::Error::UnusedParams));
}

#[test]
fn test_enum_with_lifetime() {
    #[derive(Parameterized, Debug, Clone, PartialEq, Eq)]
    enum Enum<'p, V: Parameter> {
        X(V),
        Y(&'p str, usize),
    }

    let mut value = Enum::X(42usize);
    let structure = Enum::X(Placeholder);

    assert_eq!(value.param_count(), 1);
    assert_eq!(value.param_structure(), structure);
    assert_eq!(value.params().collect::<Vec<_>>(), vec![&42usize]);
    assert_eq!(value.params_mut().collect::<Vec<_>>(), vec![&mut 42usize]);
    assert_eq!(value.clone().into_params().collect::<Vec<_>>(), vec![42usize]);
    assert_eq!(Enum::from_params(structure.clone(), vec![42i32]), Ok(Enum::X(42i32)));
    assert_eq!(Enum::from_params(structure, [0usize; 10]), Err(ryft::Error::UnusedParams));

    let mut value = Enum::Y("hey there", 42usize);
    let structure = Enum::Y("hey there", 42usize);

    assert_eq!(value.param_count(), 0);
    assert_eq!(value.param_structure(), structure);
    assert_eq!(value.params().collect::<Vec<_>>(), Vec::<&usize>::new());
    assert_eq!(value.params_mut().collect::<Vec<_>>(), Vec::<&mut usize>::new());
    assert_eq!(value.clone().into_params().collect::<Vec<_>>(), Vec::<usize>::new());
    assert_eq!(Enum::from_params(structure.clone(), Vec::<usize>::new()), Ok(value));
    assert_eq!(Enum::from_params(structure, [0usize; 10]), Err(ryft::Error::UnusedParams));
}

#[test]
fn test_complex_type() {
    #[derive(Parameterized, Debug, Clone, PartialEq, Eq)]
    struct Struct<P: Parameter> {
        p_0: P,
        p_1: P,
        np_0: (usize, usize),
        np_1: u64,
        _phantom_data: PhantomData<P>,
    }

    #[derive(Parameterized, Debug, Clone, PartialEq, Eq)]
    struct StructWithNestedStruct<P: Parameter> {
        p_0: P,
        p_1: (usize, (i32, P, i64, Struct<P>)),
        np_0: (usize, usize),
        np_1: u64,
    }

    #[derive(Parameterized, Debug, Clone, PartialEq, Eq)]
    #[ryft(crate = "ryft")]
    enum Enum<P: Parameter> {
        Nada,
        NadaUnit(),
        Parameter(P),
        ParameterTuple(P, P),
        Stuff { stuff: Vec<P>, more_stuff: (StructWithNestedStruct<P>, P) },
        StructWithNestedStruct(StructWithNestedStruct<P>, usize),
        Irrelevant(i64),
    }

    #[derive(Parameterized, Debug, Clone, PartialEq, Eq)]
    struct StructWithEnum<P: Parameter>(Enum<P>);

    let structure = StructWithEnum::<Placeholder>(Enum::Nada);
    assert_eq!(structure.param_count(), 0);
    assert_eq!(structure.param_structure(), structure);
    assert_eq!(structure.params().collect::<Vec<_>>(), Vec::<&Placeholder>::new());
    assert_eq!(StructWithEnum::from_params(structure.clone(), Vec::new()), Ok(structure));

    let structure = StructWithEnum::<Placeholder>(Enum::NadaUnit());
    assert_eq!(structure.param_count(), 0);
    assert_eq!(structure.param_structure(), structure);
    assert_eq!(structure.params().collect::<Vec<_>>(), Vec::<&Placeholder>::new());
    assert_eq!(StructWithEnum::from_params(structure.clone(), Vec::new()), Ok(structure));

    let structure = StructWithEnum(Enum::Parameter(Placeholder));
    assert_eq!(structure.param_count(), 1);
    assert_eq!(structure.param_structure(), structure);
    assert_eq!(structure.params().collect::<Vec<_>>(), vec![&Placeholder]);
    assert_eq!(
        StructWithEnum::from_params(structure.clone(), vec![42usize]),
        Ok(StructWithEnum(Enum::Parameter(42usize)))
    );

    let structure = StructWithEnum(Enum::ParameterTuple(Placeholder, Placeholder));
    assert_eq!(structure.param_count(), 2);
    assert_eq!(structure.param_structure(), structure);
    assert_eq!(structure.params().collect::<Vec<_>>(), vec![&Placeholder, &Placeholder]);
    assert_eq!(
        StructWithEnum::from_params(structure.clone(), vec![4usize, 2usize]),
        Ok(StructWithEnum(Enum::ParameterTuple(4usize, 2usize)))
    );

    let simple_struct_structure = Struct {
        p_0: Placeholder,
        p_1: Placeholder,
        np_0: (2usize, 1usize),
        np_1: 4u64,
        _phantom_data: PhantomData::<Placeholder>,
    };

    let nested_struct_structure = StructWithNestedStruct {
        p_0: Placeholder,
        p_1: (42usize, (-2i32, Placeholder, 4i64, simple_struct_structure.clone())),
        np_0: (4usize, 2usize),
        np_1: 64u64,
    };

    let structure = StructWithEnum::<Placeholder>(Enum::Stuff {
        stuff: vec![],
        more_stuff: (nested_struct_structure.clone(), Placeholder),
    });

    assert_eq!(structure.param_count(), 5);
    assert_eq!(structure.param_structure(), structure);
    assert_eq!(structure.params().collect::<Vec<_>>(), vec![&Placeholder; 5]);
    assert_eq!(
        StructWithEnum::from_params(structure.clone(), vec![0i32, 1i32, -42i32, -2i32, 0i32]),
        Ok(StructWithEnum(Enum::Stuff {
            stuff: vec![],
            more_stuff: (
                StructWithNestedStruct {
                    p_0: 0i32,
                    p_1: (
                        42usize,
                        (
                            -2i32,
                            1i32,
                            4i64,
                            Struct {
                                p_0: -42i32,
                                p_1: -2i32,
                                np_0: (2usize, 1usize),
                                np_1: 4u64,
                                _phantom_data: PhantomData::<i32>,
                            }
                        )
                    ),
                    np_0: (4usize, 2usize),
                    np_1: 64u64,
                },
                0i32,
            ),
        }))
    );

    let structure = StructWithEnum::<Placeholder>(Enum::Stuff {
        stuff: vec![Placeholder, Placeholder, Placeholder],
        more_stuff: (nested_struct_structure.clone(), Placeholder),
    });

    assert_eq!(structure.param_count(), 8);
    assert_eq!(structure.param_structure(), structure);
    assert_eq!(structure.params().collect::<Vec<_>>(), vec![&Placeholder; 8]);
    assert_eq!(
        StructWithEnum::from_params(structure.clone(), vec![-1i32, -2i32, -3i32, 0i32, 1i32, -42i32, -2i32, 0i32]),
        Ok(StructWithEnum(Enum::Stuff {
            stuff: vec![-1i32, -2i32, -3i32],
            more_stuff: (
                StructWithNestedStruct {
                    p_0: 0i32,
                    p_1: (
                        42usize,
                        (
                            -2i32,
                            1i32,
                            4i64,
                            Struct {
                                p_0: -42i32,
                                p_1: -2i32,
                                np_0: (2usize, 1usize),
                                np_1: 4u64,
                                _phantom_data: PhantomData::<i32>,
                            }
                        )
                    ),
                    np_0: (4usize, 2usize),
                    np_1: 64u64,
                },
                0i32,
            ),
        }))
    );

    let structure = StructWithEnum::<Placeholder>(Enum::StructWithNestedStruct(nested_struct_structure, 42usize));
    assert_eq!(structure.param_count(), 4);
    assert_eq!(structure.param_structure(), structure);
    assert_eq!(structure.params().collect::<Vec<_>>(), vec![&Placeholder; 4]);
    assert_eq!(
        StructWithEnum::from_params(structure.clone(), vec![0i32, 1i32, -42i32, -2i32]),
        Ok(StructWithEnum(Enum::StructWithNestedStruct(
            StructWithNestedStruct {
                p_0: 0i32,
                p_1: (
                    42usize,
                    (
                        -2i32,
                        1i32,
                        4i64,
                        Struct {
                            p_0: -42i32,
                            p_1: -2i32,
                            np_0: (2usize, 1usize),
                            np_1: 4u64,
                            _phantom_data: PhantomData::<i32>,
                        }
                    )
                ),
                np_0: (4usize, 2usize),
                np_1: 64u64,
            },
            42usize
        )))
    );

    let structure = StructWithEnum::<Placeholder>(Enum::Irrelevant(42i64));
    assert_eq!(structure.param_count(), 0);
    assert_eq!(structure.param_structure(), structure);
    assert_eq!(structure.params().collect::<Vec<_>>(), Vec::<&Placeholder>::new());
    assert_eq!(StructWithEnum::from_params(structure.clone(), Vec::new()), Ok(structure));
}

#[test]
fn test_nested_vec() {
    #[derive(Parameterized, Debug, Clone, PartialEq, Eq)]
    struct VecWrapper<P: Parameter>(Vec<P>);

    let mut value = VecWrapper(vec![4usize, 2usize]);
    let structure = VecWrapper(vec![Placeholder, Placeholder]);
    let insufficient_params_error = Err(ryft::Error::InsufficientParams { expected_count: 2 });

    assert_eq!(value.param_count(), 2);
    assert_eq!(value.param_structure(), structure);
    assert_eq!(value.params().collect::<Vec<_>>(), vec![&4usize, &2usize]);
    assert_eq!(value.params_mut().collect::<Vec<_>>(), vec![&mut 4usize, &mut 2usize]);
    assert_eq!(value.clone().into_params().collect::<Vec<_>>(), vec![4usize, 2usize]);
    assert_eq!(VecWrapper::from_params(structure.clone(), Vec::<usize>::new()), insufficient_params_error);
    assert_eq!(VecWrapper::from_params(structure.clone(), vec![4i64, 2i64]), Ok(VecWrapper(vec![4i64, 2i64])));
    assert_eq!(VecWrapper::from_params(structure, [0usize; 10]), Err(ryft::Error::UnusedParams));

    #[derive(Parameterized, Debug, Clone, PartialEq, Eq)]
    struct MultiVecWrapper<P: Parameter>(Vec<P>, usize, (Vec<P>, Vec<Vec<P>>));

    let mut value = MultiVecWrapper(vec![4usize, 2usize], 0usize, (Vec::new(), vec![vec![42usize]]));
    let structure = MultiVecWrapper(vec![Placeholder, Placeholder], 0usize, (Vec::new(), vec![vec![Placeholder]]));
    let insufficient_params_error = Err(ryft::Error::InsufficientParams { expected_count: 3 });

    assert_eq!(value.param_count(), 3);
    assert_eq!(value.param_structure(), structure);
    assert_eq!(value.params().collect::<Vec<_>>(), vec![&4usize, &2usize, &42usize]);
    assert_eq!(value.params_mut().collect::<Vec<_>>(), vec![&mut 4usize, &mut 2usize, &mut 42usize]);
    assert_eq!(value.clone().into_params().collect::<Vec<_>>(), vec![4usize, 2usize, 42usize]);
    assert_eq!(MultiVecWrapper::from_params(structure.clone(), Vec::<usize>::new()), insufficient_params_error);
    assert_eq!(
        MultiVecWrapper::from_params(structure.clone(), vec![4i64, 2i64, 42i64]),
        Ok(MultiVecWrapper(vec![4i64, 2i64], 0usize, (Vec::new(), vec![vec![42i64]])))
    );
    assert_eq!(MultiVecWrapper::from_params(structure, [0usize; 10]), Err(ryft::Error::UnusedParams));

    value.params_mut().for_each(|param| *param *= 2);
    assert_eq!(value, MultiVecWrapper(vec![8usize, 4usize], 0usize, (Vec::new(), vec![vec![84usize]])));
}

#[test]
fn test_nested_vec_of_tuples() {
    #[derive(Parameterized, Debug, Clone, PartialEq, Eq)]
    struct NestedVecOfTuples<P: Parameter>(Vec<(P, P)>, usize, (Vec<P>, Vec<(Vec<P>, P)>));

    let mut value = NestedVecOfTuples(vec![(4i8, 2i8), (2i8, 1i8)], 0usize, (Vec::new(), vec![(vec![42i8], 0i8)]));
    let structure = NestedVecOfTuples(
        vec![(Placeholder, Placeholder), (Placeholder, Placeholder)],
        0usize,
        (Vec::new(), vec![(vec![Placeholder], Placeholder)]),
    );
    let insufficient_params_error = Err(ryft::Error::InsufficientParams { expected_count: 6 });

    assert_eq!(value.param_count(), 6);
    assert_eq!(value.param_structure(), structure);
    assert_eq!(value.params().collect::<Vec<_>>(), vec![&4i8, &2i8, &2i8, &1i8, &42i8, &0i8]);
    assert_eq!(
        value.params_mut().collect::<Vec<_>>(),
        vec![&mut 4i8, &mut 2i8, &mut 2i8, &mut 1i8, &mut 42i8, &mut 0i8]
    );
    assert_eq!(value.clone().into_params().collect::<Vec<_>>(), vec![4i8, 2i8, 2i8, 1i8, 42i8, 0i8]);
    assert_eq!(NestedVecOfTuples::from_params(structure.clone(), Vec::<usize>::new()), insufficient_params_error);
    assert_eq!(
        NestedVecOfTuples::from_params(structure.clone(), vec![4i64, 2i64, 2i64, 1i64, 42i64, 0i64]),
        Ok(NestedVecOfTuples(vec![(4i64, 2i64), (2i64, 1i64)], 0usize, (Vec::new(), vec![(vec![42i64], 0i64)])))
    );
    assert_eq!(NestedVecOfTuples::from_params(structure, [0usize; 10]), Err(ryft::Error::UnusedParams));

    value.params_mut().for_each(|param| *param *= 2);
    assert_eq!(value, NestedVecOfTuples(vec![(8i8, 4i8), (4i8, 2i8)], 0usize, (Vec::new(), vec![(vec![84i8], 0i8)])));
}

#[test]
fn test_generics() {
    #[derive(Parameterized, Debug, Clone, PartialEq, Eq)]
    struct Struct<P: Parameter, V> {
        p_0: P,
        p_1: P,
        np_0: (V, V),
        np_1: u64,
    }

    let np_0 = (4usize, 2usize);
    let np_1 = 42u64;
    let mut value = Struct { p_0: 4usize, p_1: 2usize, np_0, np_1 };
    let structure = Struct { p_0: Placeholder, p_1: Placeholder, np_0, np_1 };
    let insufficient_params_error = Err(ryft::Error::InsufficientParams { expected_count: 2 });

    assert_eq!(value.param_count(), 2);
    assert_eq!(value.param_structure(), structure);
    assert_eq!(value.params().collect::<Vec<_>>(), vec![&4usize, &2usize]);
    assert_eq!(value.params_mut().collect::<Vec<_>>(), vec![&mut 4usize, &mut 2usize]);
    assert_eq!(value.clone().into_params().collect::<Vec<_>>(), vec![4usize, 2usize]);
    assert_eq!(Struct::from_params(structure.clone(), Vec::<usize>::new()), insufficient_params_error);
    assert_eq!(
        Struct::from_params(structure.clone(), vec![4i64, 2i64]),
        Ok(Struct { p_0: 4i64, p_1: 2i64, np_0, np_1 })
    );
    assert_eq!(Struct::from_params(structure, [0usize; 10]), Err(ryft::Error::UnusedParams));

    value.params_mut().for_each(|param| *param *= 2);
    assert_eq!(value, Struct { p_0: 8usize, p_1: 4usize, np_0, np_1 })
}

#[test]
fn test_map_params() {
    // This test implements a toy forward-differentiation approach to test [Parameterized::map_params].

    #[derive(Parameter, Parameterized, Debug, Clone, PartialEq, Eq)]
    struct ValueWithTangent<P: Parameter> {
        primal: P,
        tangent: P,
    }

    impl<P: Parameter + Add<Output = P>> std::ops::Add for ValueWithTangent<P> {
        type Output = ValueWithTangent<P>;

        fn add(self, rhs: ValueWithTangent<P>) -> Self::Output {
            ValueWithTangent { primal: self.primal + rhs.primal, tangent: self.tangent + rhs.tangent }
        }
    }

    impl<P: Clone + Parameter + Add<Output = P> + Mul<Output = P>> std::ops::Mul for ValueWithTangent<P> {
        type Output = ValueWithTangent<P>;

        fn mul(self, rhs: ValueWithTangent<P>) -> Self::Output {
            ValueWithTangent {
                primal: self.primal.clone() * rhs.primal.clone(),
                tangent: self.primal * rhs.tangent + self.tangent * rhs.primal,
            }
        }
    }

    #[derive(Parameterized)]
    struct Linear<P: Parameter> {
        weights: P,
        bias: P,
    }

    impl<P: Clone + Parameter + Add<Output = P> + Mul<Output = P>> Linear<P> {
        fn call(&self, x: P) -> P {
            self.weights.clone() * x + self.bias.clone()
        }
    }

    #[derive(Parameterized)]
    struct MLP<P: Parameter, const N: usize> {
        layers: [Linear<P>; N],
    }

    impl<P: Clone + Parameter + Add<Output = P> + Mul<Output = P>, const N: usize> MLP<P, N> {
        fn call(&self, x: P) -> P {
            self.layers.iter().fold(x, |result, layer| layer.call(result))
        }
    }

    fn f<T: Clone + Add<Output = T> + Mul<Output = T>>(x: T, y: T) -> T {
        x.clone() + (x.clone() * x * y)
    }

    // We will go a little weird and compute a second derivative which means we need two layers
    // of [ValueWithTangent] wrapping.
    let x = ValueWithTangent { primal: 1f64, tangent: 1f64 };
    let x = ValueWithTangent { primal: x.clone(), tangent: x };
    let linear1 = Linear { weights: 4f64, bias: 2f64 };
    let linear2 = Linear { weights: -1f64, bias: -2f64 };
    let mlp = MLP { layers: [linear1, linear2] };
    let mlp = mlp.map_params(|p| ValueWithTangent { primal: p, tangent: 1f64 }).unwrap();
    let mlp = mlp.map_params(|p| ValueWithTangent { primal: p.clone(), tangent: p }).unwrap();

    assert_eq!(
        mlp.call(x.clone()),
        ValueWithTangent {
            primal: ValueWithTangent { primal: -8.0, tangent: 1.0 },
            tangent: ValueWithTangent { primal: -18.0, tangent: 0.0 }
        }
    );

    assert_eq!(
        f(x.clone(), x),
        ValueWithTangent {
            primal: ValueWithTangent { primal: 2.0, tangent: 4.0 },
            tangent: ValueWithTangent { primal: 4.0, tangent: 10.0 }
        }
    );
}

#[test]
fn test_errors() {
    let test_cases = trybuild::TestCases::new();
    test_cases.compile_fail("tests/parameters/error_attributes.rs");
    test_cases.compile_fail("tests/parameters/error_enums.rs");
    test_cases.compile_fail("tests/parameters/error_no_parameters.rs");
    test_cases.compile_fail("tests/parameters/error_references.rs");
    test_cases.compile_fail("tests/parameters/error_self.rs");
    test_cases.compile_fail("tests/parameters/error_structs.rs");
    test_cases.compile_fail("tests/parameters/error_tuples.rs");
    test_cases.compile_fail("tests/parameters/error_unions.rs");
}
