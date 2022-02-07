use numpy_type_desc::{dtype, td, units, Element as _, Record, Timedelta64};

use pyo3::{py_run, PyObject};

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
    #[repr(C)]
    struct Foo<T, const N: usize> {
        x: [[T; N]; 3],
        y: (PyObject, Timedelta64<units::Nanosecond>),
    }

    assert_eq!(
        Foo::<i64, 2>::type_descriptor(),
        td!({"x":0 => [(3, 2); <i64], "y":48 => {0 => O, 8 => m8[ns] [16, 8]} [64, 8]})
    );

    pyo3::Python::with_gil(|py| {
        let dt = dtype::<Foo<i64, 2>>(py).unwrap();
        py_run!(
            py,
            dt,
            r#"
            args = dict(
                names=['x', 'y'],
                offsets=[0, 48],
                formats=[('<i8', (3, 2)),[('f0', 'O'), ('f1', '<m8[ns]')]],
                itemsize=64,
                aligned=True,
            )
            dtype = __import__('numpy').dtype(args)
            assert dtype == dt
        "#
        );
    });
}
