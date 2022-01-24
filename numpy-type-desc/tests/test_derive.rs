use numpy_type_desc::{td, units, Element as _, Record, Timedelta64};

use pyo3::PyObject;

#[test]
fn test_struct_1() {
    #[derive(Clone, Record)]
    struct Foo {
        x: bool,
        y: i64,
    }

    assert_eq!(Foo::type_descriptor(), td!({"x":8 => bool, "y":0 => <i64 [16, 8]}));
}

#[test]
fn test_struct_2() {
    #[derive(Clone, Record)]
    #[repr(packed)]
    struct Foo<T, const N: usize> {
        x: [[T; N]; 3],
        y: (PyObject, Timedelta64<units::Nanosecond>),
    }

    assert_eq!(
        Foo::<i64, 2>::type_descriptor(),
        td!({"x":0 => [(3, 2); <i64], "y":48 => {0 => object, 8 => timedelta64[ns] [16, 8]} [64, 1]})
    );
}
