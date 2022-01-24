use std::mem;
use std::os::raw;
use std::ptr;

use pyo3::{
    exceptions::{PyMemoryError, PyValueError},
    ffi::{Py_DECREF, Py_XDECREF},
    types::{PyDict, PyTuple},
    AsPyPointer, IntoPy, IntoPyPointer, Py, PyErr, PyResult, Python,
};

use numpy::npyffi::{
    PyArray_DatetimeDTypeMetaData, PyArray_DatetimeMetaData, PyArray_Descr, PyArray_malloc,
    NPY_ALIGNED_STRUCT, NPY_BYTEORDER_CHAR, NPY_FROM_FIELDS, NPY_NEEDS_PYAPI, NPY_TYPES,
    PY_ARRAY_API,
};
use numpy::PyArrayDescr;

use pyo3_type_desc::{
    ArrayDescriptor, ComplexSize, Endian, FloatSize, IntegerSize, RecordDescriptor,
    Scalar as BaseScalar, Signedness, TypeDescriptor,
};

use crate::datetime::DatetimeUnit;
use crate::element::{ArrayElement, Scalar};

#[inline]
unsafe fn pyarray_descr_replace(descr: &mut *mut PyArray_Descr) {
    // an equivalent of PyArray_DESCR_REPLACE macro
    let new = PY_ARRAY_API.PyArray_DescrNew(*descr);
    Py_XDECREF(*descr as _);
    *descr = new;
}

fn npy_int_type_lookup<T, T0, T1, T2>(npy_types: [NPY_TYPES; 3]) -> NPY_TYPES {
    // `npy_common.h` defines the integer aliases. In order, it checks:
    // NPY_BITSOF_LONG, NPY_BITSOF_LONGLONG, NPY_BITSOF_INT, NPY_BITSOF_SHORT, NPY_BITSOF_CHAR
    // and assigns the alias to the first matching size, so we should check in this order.
    match mem::size_of::<T>() {
        x if x == mem::size_of::<T0>() => npy_types[0],
        x if x == mem::size_of::<T1>() => npy_types[1],
        x if x == mem::size_of::<T2>() => npy_types[2],
        _ => panic!("Unable to match integer type descriptor: {:?}", npy_types),
    }
}

#[inline]
fn checked_elsize(
    size: usize, itemsize: usize, err: impl FnOnce() -> PyErr,
) -> PyResult<raw::c_int> {
    if let Some(elsize) = size.checked_mul(itemsize) {
        if elsize <= raw::c_int::MAX as usize {
            return Ok(elsize as _);
        }
    }
    Err(err())
}

unsafe fn create_dtype_object(py: Python) -> PyResult<*mut PyArray_Descr> {
    let dtype = PY_ARRAY_API.PyArray_DescrFromType(NPY_TYPES::NPY_OBJECT as _);
    if dtype.is_null() {
        return Err(PyErr::fetch(py));
    }
    Ok(dtype)
}

unsafe fn create_dtype_scalar(py: Python, scalar: &Scalar) -> PyResult<*mut PyArray_Descr> {
    #[derive(Clone, Copy)]
    pub enum Quirks {
        None,
        Flexible(usize, usize),
        Endian(Endian),
        DatetimeUnit(DatetimeUnit),
    }

    type B = BaseScalar;
    type S = Signedness;
    type I = IntegerSize;
    type F = FloatSize;
    type C = ComplexSize;
    type N = NPY_TYPES;
    type Q = Quirks;

    let (npy_type, quirks) = match *scalar {
        Scalar::Base(scalar) => match scalar {
            B::Bool => (N::NPY_BOOL, Q::None),
            B::Char(S::Signed) => (N::NPY_BYTE, Q::None),
            B::Char(S::Unsigned) => (N::NPY_UBYTE, Q::None),
            B::Integer(I::Two, S::Signed, endian) => (N::NPY_SHORT, Q::Endian(endian)),
            B::Integer(I::Two, S::Unsigned, endian) => (N::NPY_USHORT, Q::Endian(endian)),
            B::Integer(I::Four, S::Signed, endian) => (
                npy_int_type_lookup::<i32, raw::c_long, raw::c_int, raw::c_short>([
                    N::NPY_LONG,
                    N::NPY_INT,
                    N::NPY_SHORT,
                ]),
                Q::Endian(endian),
            ),
            B::Integer(I::Four, S::Unsigned, endian) => (
                npy_int_type_lookup::<u32, raw::c_ulong, raw::c_uint, raw::c_ushort>([
                    N::NPY_ULONG,
                    N::NPY_UINT,
                    N::NPY_USHORT,
                ]),
                Q::Endian(endian),
            ),
            B::Integer(I::Eight, S::Signed, endian) => (
                npy_int_type_lookup::<i64, raw::c_long, raw::c_longlong, raw::c_int>([
                    N::NPY_LONG,
                    N::NPY_LONGLONG,
                    N::NPY_INT,
                ]),
                Q::Endian(endian),
            ),
            B::Integer(I::Eight, S::Unsigned, endian) => (
                npy_int_type_lookup::<u64, raw::c_ulong, raw::c_ulonglong, raw::c_uint>([
                    N::NPY_ULONG,
                    N::NPY_ULONGLONG,
                    N::NPY_UINT,
                ]),
                Q::Endian(endian),
            ),
            B::Float(F::Two, endian) => (N::NPY_HALF, Q::Endian(endian)),
            B::Float(F::Four, endian) => (N::NPY_FLOAT, Q::Endian(endian)),
            B::Float(F::Eight, endian) => (N::NPY_DOUBLE, Q::Endian(endian)),
            B::Complex(C::Four, endian) => (N::NPY_CFLOAT, Q::Endian(endian)),
            B::Complex(C::Eight, endian) => (N::NPY_CDOUBLE, Q::Endian(endian)),
            B::Bytes(size) => (N::NPY_STRING, Q::Flexible(size, 1)),
            B::Ucs4(size) => (N::NPY_UNICODE, Q::Flexible(size, 4)), // TODO: check if correct
        },
        Scalar::Datetime(unit) => (N::NPY_DATETIME, Q::DatetimeUnit(unit)),
        Scalar::Timedelta(unit) => (N::NPY_TIMEDELTA, Q::DatetimeUnit(unit)),
    };

    let mut dtype = PY_ARRAY_API.PyArray_DescrFromType(npy_type as _);
    if dtype.is_null() {
        return Err(PyErr::fetch(py));
    }

    match quirks {
        Quirks::Flexible(size, width) => {
            (*dtype).elsize = checked_elsize(size, width, || {
                PyValueError::new_err(
                    "invalid size of a string type: itemsize does not fit into a C int.",
                )
            })?;
        }
        Quirks::Endian(endian) if endian.is_swapped() => {
            // we bypass PyArray_DescrNewByteorder since it has lots of extra logic
            // that we don't need (e.g. recursively setting endianness for subarrays
            // and record type fields)
            pyarray_descr_replace(&mut dtype);
            if dtype.is_null() {
                return Err(PyErr::fetch(py));
            }
            (*dtype).byteorder = NPY_BYTEORDER_CHAR::NPY_OPPBYTE as _;
        }
        Quirks::DatetimeUnit(unit) => {
            // reference: create_datetime_dtype() in descriptor.c
            let meta = PyArray_DatetimeMetaData { base: unit.into_npy_datetimeunit(), num: 1 };
            // pyarray_descr_replace is not required here, we can just mutate the dtype
            let dtype_meta = (*dtype).c_metadata as *mut PyArray_DatetimeDTypeMetaData;
            (*dtype_meta).meta = meta;
        }
        _ => {}
    }

    Ok(dtype)
}

unsafe fn create_dtype_array(
    py: Python, arr: &ArrayDescriptor<Scalar>,
) -> PyResult<*mut PyArray_Descr> {
    // reference: _convert_from_tuple() in descriptor.c
    let ArrayDescriptor { desc, shape } = arr;

    let nbytes = checked_elsize(arr.itemsize(), arr.size(), || {
        PyValueError::new_err(
            "invalid shape in fixed-type tuple: dtype size in bytes must fit into a C int.",
        )
    })?;

    let base = create_dtype_any(py, desc)?;

    let dtype = PY_ARRAY_API.PyArray_DescrFromType(NPY_TYPES::NPY_VOID as _);
    if dtype.is_null() {
        return Err(PyErr::fetch(py));
    }

    (*dtype).subarray = PyArray_malloc(mem::size_of::<PyArray_Descr>()) as _;
    if (*dtype).subarray.is_null() {
        Py_DECREF(dtype as _);
        return Err(PyMemoryError::new_err(""));
    }

    (*dtype).elsize = nbytes as _;
    (*dtype).flags = (*base).flags;
    (*dtype).alignment = (*base).alignment;
    (*(*dtype).subarray).base = base;
    Py_XDECREF((*dtype).fields);
    Py_XDECREF((*dtype).names);
    (*dtype).fields = ptr::null_mut() as _;
    (*dtype).names = ptr::null_mut() as _;
    (*(*dtype).subarray).shape = PyTuple::new(py, shape.iter().copied()).into_ptr();

    Ok(dtype)
}

unsafe fn create_dtype_record(
    py: Python, rec: &RecordDescriptor<Scalar>,
) -> PyResult<*mut PyArray_Descr> {
    // reference: _convert_from_dict() in descriptor.c
    let RecordDescriptor { fields, itemsize, alignment } = rec;

    let mut flags = NPY_NEEDS_PYAPI;
    if rec.is_aligned() {
        // we will take the initiative to always mark aligned structs as 'aligned' because we can
        flags |= NPY_ALIGNED_STRUCT;
    }

    // TODO: validate layout:
    // - itemsize is a multiple of alignment
    // - no repeating field names
    // - no invalid (empty,\0,\n) field names
    // - no field overlaps (numpy only checks it for objects)
    // - optional: non-zero size?
    // - optional: at least on field?

    let names = rec.fill_names().map(|name| name.to_string()).collect::<Vec<_>>();
    let names_tuple = PyTuple::new(py, &names);

    let fields_dict = PyDict::new(py);
    for (field, name) in fields.iter().zip(&names) {
        let dtype: Py<PyArrayDescr> = dtype_from_type_descriptor(py, &field.desc)?.into_py(py);
        flags |= (*(dtype.as_ptr() as *mut PyArray_Descr)).flags & NPY_FROM_FIELDS;
        let field_item: Py<PyTuple> = (dtype, field.offset).into_py(py);
        if fields_dict.contains(&name).unwrap_or(false) {
            return Err(PyValueError::new_err(format!(
                "name {:?} already used as a name or title",
                name
            )));
        }
        fields_dict.set_item(name, field_item)?;
    }

    let dtype = PY_ARRAY_API.PyArray_DescrFromType(NPY_TYPES::NPY_VOID as _);
    if dtype.is_null() {
        return Err(PyErr::fetch(py));
    }

    (*dtype).fields = fields_dict.into_ptr();
    (*dtype).names = names_tuple.into_ptr();
    (*dtype).flags = flags;
    // TODO: numpy assigns alignment only in 'aligned' mode, should we just always set it?
    (*dtype).alignment = alignment.unwrap_or(1) as _;
    (*dtype).elsize = *itemsize as _;

    Ok(dtype)
}

unsafe fn create_dtype_any(
    py: Python, desc: &TypeDescriptor<Scalar>,
) -> PyResult<*mut PyArray_Descr> {
    match desc {
        TypeDescriptor::Object => create_dtype_object(py),
        TypeDescriptor::Scalar(scalar) => create_dtype_scalar(py, scalar),
        TypeDescriptor::Array(arr) => create_dtype_array(py, arr),
        TypeDescriptor::Record(rec) => create_dtype_record(py, rec),
    }
}

pub fn dtype_from_type_descriptor<'py>(
    py: Python<'py>, desc: &TypeDescriptor<Scalar>,
) -> PyResult<&'py PyArrayDescr> {
    Ok(unsafe { py.from_owned_ptr(create_dtype_any(py, desc)? as _) })
}

pub fn dtype<T: ArrayElement>(py: Python) -> PyResult<&PyArrayDescr> {
    dtype_from_type_descriptor(py, &T::type_descriptor())
}

#[cfg(test)]
mod tests {
    use std::mem;

    use memoffset::offset_of_tuple;
    use num_complex::{Complex32, Complex64};
    use numpy::PyArrayDescr;
    use pyo3::{PyObject, Python};

    use crate::{dtype, dtype_from_type_descriptor, td, units, Datetime64, Element, Timedelta64};

    #[test]
    fn test_object() {
        Python::with_gil(|py| {
            let d = dtype_from_type_descriptor(py, &PyObject::type_descriptor()).unwrap();
            assert_eq!(d, dtype::<PyObject>(py).unwrap());
            assert_eq!(d.itemsize(), mem::size_of::<PyObject>());
            assert_eq!(d.alignment(), mem::align_of::<PyObject>());
            assert_eq!(d.typeobj().name().unwrap(), "object_");
            assert!(d.has_object());
        })
    }

    #[test]
    fn test_base_scalars() {
        macro_rules! check {
            ($py:expr, $ty:ty, $str:expr, $($tt:tt)+) => {{
                let desc = <$ty>::type_descriptor();
                assert_eq!(desc, td!($($tt)+));
                let d = dtype_from_type_descriptor($py, &desc).unwrap();
                assert_eq!(d, dtype::<$ty>($py).unwrap());
                assert_eq!(d.itemsize(), mem::size_of::<$ty>());
                assert_eq!(d.alignment(), mem::align_of::<$ty>());
                check!($py, $str, $($tt)+);
            }};
            ($py:expr, $str:expr, $($tt:tt)+) => {{
                let desc = td!($($tt)+);
                let d = dtype_from_type_descriptor($py, &desc).unwrap();
                assert_eq!(d.typeobj().name().unwrap(), $str);
                assert_eq!(d.has_object(), $str == "object_");
                assert!(!d.has_subarray());
                assert!(d.names().is_none());
            }};
        }
        Python::with_gil(|py| {
            check!(py, "object_", O);
            check!(py, bool, "bool_", ?);
            check!(py, i8, "int8", i8);
            check!(py, u8, "uint8", u8);
            check!(py, i16, "int16", =i16);
            check!(py, u16, "uint16", =u16);
            check!(py, i32, "int32", =i32);
            check!(py, u32, "uint32", =u32);
            check!(py, i64, "int64", =i64);
            check!(py, u64, "uint64", =u64);
            check!(py, f32, "float32", =f32);
            check!(py, f64, "float64", =f64);
            check!(py, Complex32, "complex64", =c32);
            check!(py, Complex64, "complex128", =c64);
            #[cfg(target_pointer_width = "32")]
            {
                check!(py, usize, "uint32", =u32);
                check!(py, isize, "int32", =i32);
            }
            #[cfg(target_pointer_width = "64")]
            {
                check!(py, usize, "uint64", =u64);
                check!(py, isize, "int64", =i64);
            }
        });
    }

    #[test]
    fn test_datetime_scalars() {
        macro_rules! check {
            ($py:expr, $np_ty:ident, $unit:ident, $ty:ty) => {{
                let desc = td!($np_ty[$unit]);
                let d = dtype_from_type_descriptor($py, &desc).unwrap();
                let type_str = stringify!($np_ty);
                assert_eq!(d.typeobj().name().unwrap(), type_str);
                assert!(!d.has_object());
                assert_eq!(d.itemsize(), 8);
                assert_eq!(d.alignment(), 8);
                assert_eq!(dtype::<$ty>($py).unwrap(), d);
            }};
        }
        Python::with_gil(|py| {
            check!(py, datetime64, Y, Datetime64<units::Year>);
            check!(py, datetime64, M, Datetime64<units::Month>);
            check!(py, datetime64, W, Datetime64<units::Week>);
            check!(py, datetime64, D, Datetime64<units::Day>);
            check!(py, datetime64, h, Datetime64<units::Hour>);
            check!(py, datetime64, m, Datetime64<units::Minute>);
            check!(py, datetime64, s, Datetime64<units::Second>);
            check!(py, datetime64, ms, Datetime64<units::Millisecond>);
            check!(py, datetime64, us, Datetime64<units::Microsecond>);
            check!(py, datetime64, ns, Datetime64<units::Nanosecond>);
            check!(py, datetime64, ps, Datetime64<units::Picosecond>);
            check!(py, datetime64, fs, Datetime64<units::Femtosecond>);
            check!(py, datetime64, as, Datetime64<units::Attosecond>);

            check!(py, timedelta64, Y, Timedelta64<units::Year>);
            check!(py, timedelta64, M, Timedelta64<units::Month>);
            check!(py, timedelta64, W, Timedelta64<units::Week>);
            check!(py, timedelta64, D, Timedelta64<units::Day>);
            check!(py, timedelta64, h, Timedelta64<units::Hour>);
            check!(py, timedelta64, m, Timedelta64<units::Minute>);
            check!(py, timedelta64, s, Timedelta64<units::Second>);
            check!(py, timedelta64, ms, Timedelta64<units::Millisecond>);
            check!(py, timedelta64, us, Timedelta64<units::Microsecond>);
            check!(py, timedelta64, ns, Timedelta64<units::Nanosecond>);
            check!(py, timedelta64, ps, Timedelta64<units::Picosecond>);
            check!(py, timedelta64, fs, Timedelta64<units::Femtosecond>);
            check!(py, timedelta64, as, Timedelta64<units::Attosecond>);
        });
    }

    #[test]
    fn test_tuple() {
        fn check_field(parent: &PyArrayDescr, name: &str, d: &PyArrayDescr, offset: usize) {
            let field = parent.get_field(name).unwrap();
            assert_eq!(field, (d, offset));
        }

        macro_rules! check {
            ($py:expr, ($($i:tt : [$ty:ty] => [$($tt:tt)+]),+)) => {{
                type T = ($($ty,)+);
                let d = dtype_from_type_descriptor($py, &T::type_descriptor()).unwrap();
                assert_eq!(d.names().unwrap(), &[$(format!("f{}", $i),)+]);
                $(
                let field_name = format!("f{}", $i);
                let field_dtype = dtype_from_type_descriptor($py, &<$ty>::type_descriptor()).unwrap();
                let field_offset = offset_of_tuple!(T, $i);
                check_field(d, &field_name, field_dtype, field_offset);
                )+
                assert!(d.is_aligned_struct());
            }};
            (@last $head:tt $($tail:tt)+) => { check!(@last $($tail)+) };
            (@last $tt:tt) => { $tt };
        }

        Python::with_gil(|py| {
            type X = (bool, u64);
            let (x0, x1) = (offset_of_tuple!(X, 0), offset_of_tuple!(X, 1));
            let (x_s, x_a) = (mem::size_of::<X>(), mem::align_of::<X>());
            type Y = (i8, X, PyObject);
            let (y0, y1, y2) =
                (offset_of_tuple!(Y, 0), offset_of_tuple!(Y, 1), offset_of_tuple!(Y, 2));
            let (y_s, y_a) = (mem::size_of::<Y>(), mem::align_of::<Y>());
            assert_eq!(
                Y::type_descriptor(),
                td!({y0 => i8, y1 => {x0 => ?, x1 => u64 [x_s, x_a]}, y2 => object [y_s, y_a]})
            );
            assert!(!dtype::<X>(py).unwrap().has_object());
            assert!(dtype::<Y>(py).unwrap().has_object());

            check!(py, (0: [u64] => [u64]));
            check!(py, (0: [i32] => [=i32], 1: [bool] => [?]));
            check!(py, (
                0: [bool] => [?],
                1: [PyObject] => [O],
                2: [Complex32] => [c32],
                3: [Complex64] => [c64],
                4: [f32] => [f32],
                5: [f64] => [f64],
                6: [i8] => [i8],
                7: [u8] => [u8],
                8: [i16] => [=i16],
                9: [u32] => [u32],
                10: [i64] => [=i64],
                11: [u64] => [u64]
            ));
        });
    }

    #[rustversion::since(1.51)]
    #[test]
    fn test_array() {
        macro_rules! check {
            ($py:expr, $ty:ty, [$($tt:tt)+], $shape:expr, $base:ty) => {{
                let desc = td!([$($tt)+]);
                let d = dtype_from_type_descriptor($py, &desc).unwrap();
                assert_eq!(d, dtype::<$ty>($py).unwrap());
                assert!(d.has_subarray());
                assert_eq!(d.base(), dtype::<$base>($py).unwrap());
                assert_eq!(d.shape(), $shape);
                assert_eq!(d.ndim(), $shape.len());
            }};
        }
        Python::with_gil(|py| {
            check!(py, [u8; 3], [(3); u8], vec![3], u8);
            check!(py, [u64; 0], [(0); u64], vec![0], u64);
            check!(py, [[bool; 3]; 4], [(4, 3); bool], vec![4, 3], bool);
        });
    }
}
