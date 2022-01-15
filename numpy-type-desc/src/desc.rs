use std::fmt::{self, Debug};
use std::mem;
use std::ops::Deref;
use std::os::raw;

use pyo3::{ffi::Py_DECREF, PyResult, Python};

use numpy::npyffi::{NPY_TYPES, PY_ARRAY_API};
use numpy::PyArrayDescr;

use pyo3_type_desc::{
    ArrayDescriptor, ComplexSize, Element, Endian, FloatSize, IntegerSize, RecordDescriptor,
    Scalar as BaseScalar, ScalarDescriptor, Signedness, TypeDescriptor,
};

use crate::datetime::DatetimeUnit;
use crate::element::Scalar;
use crate::npyffi::{PyArray_DatetimeDTypeMetaData, PyArray_DatetimeMetaData, NPY_BYTEORDER_CHAR};

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

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct NumpyDescriptor(TypeDescriptor<Scalar>);

impl<S: ScalarDescriptor + Into<Scalar>> From<TypeDescriptor<S>> for NumpyDescriptor {
    fn from(desc: TypeDescriptor<S>) -> Self {
        Self(desc.map(|scalar| scalar.clone().into()))
    }
}

impl Deref for NumpyDescriptor {
    type Target = TypeDescriptor<Scalar>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Debug for NumpyDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

impl NumpyDescriptor {
    pub fn of<T: Element<Scalar>>() -> Self {
        T::type_descriptor().into()
    }

    pub fn get_dtype_args(&self) -> DtypeArgs<'_> {
        type B = BaseScalar;
        type D<'a> = DtypeArgs<'a>;
        type S = Signedness;
        type I = IntegerSize;
        type F = FloatSize;
        type C = ComplexSize;
        type N = NPY_TYPES;

        match &self.0 {
            TypeDescriptor::Object => D::typenum(N::NPY_OBJECT),
            TypeDescriptor::Scalar(scalar) => {
                match *scalar {
                    Scalar::Base(scalar) => match scalar {
                        B::Bool => D::typenum(N::NPY_BOOL),
                        B::Char(S::Signed) => D::typenum(N::NPY_BYTE),
                        B::Char(S::Unsigned) => D::typenum(N::NPY_UBYTE),
                        B::Integer(I::Two, S::Signed, endian) => {
                            D::typenum_endian(N::NPY_SHORT, endian)
                        }
                        B::Integer(I::Two, S::Unsigned, endian) => {
                            D::typenum_endian(N::NPY_USHORT, endian)
                        }
                        B::Integer(I::Four, S::Signed, endian) => {
                            D::typenum_endian(
                                npy_int_type_lookup::<i32, raw::c_long, raw::c_int, raw::c_short>(
                                    [N::NPY_LONG, N::NPY_INT, N::NPY_SHORT],
                                ),
                                endian,
                            )
                        }
                        B::Integer(I::Four, S::Unsigned, endian) => {
                            D::typenum_endian(
                                npy_int_type_lookup::<u32, raw::c_ulong, raw::c_uint, raw::c_ushort>(
                                    [N::NPY_ULONG, N::NPY_UINT, N::NPY_USHORT],
                                ),
                                endian,
                            )
                        }
                        B::Integer(I::Eight, S::Signed, endian) => {
                            D::typenum_endian(
                                npy_int_type_lookup::<i64, raw::c_long, raw::c_longlong, raw::c_int>(
                                    [N::NPY_LONG, N::NPY_LONGLONG, N::NPY_INT],
                                ),
                                endian,
                            )
                        }
                        B::Integer(I::Eight, S::Unsigned, endian) => D::typenum_endian(
                            npy_int_type_lookup::<u64, raw::c_ulong, raw::c_ulonglong, raw::c_uint>(
                                [N::NPY_ULONG, N::NPY_ULONGLONG, N::NPY_UINT],
                            ),
                            endian,
                        ),
                        B::Float(F::Two, endian) => D::typenum_endian(N::NPY_HALF, endian),
                        B::Float(F::Four, endian) => D::typenum_endian(N::NPY_FLOAT, endian),
                        B::Float(F::Eight, endian) => D::typenum_endian(N::NPY_DOUBLE, endian),
                        B::Complex(C::Four, endian) => D::typenum_endian(N::NPY_CFLOAT, endian),
                        B::Complex(C::Eight, endian) => D::typenum_endian(N::NPY_CDOUBLE, endian),
                        B::Bytes(size) => D::typenum_flexible(N::NPY_STRING, size),
                        B::Ucs4(size) => D::typenum_flexible(N::NPY_UNICODE, size),
                    },
                    Scalar::Datetime(unit) => D::typenum_datetimeunit(N::NPY_DATETIME, unit),
                    Scalar::Timedelta(unit) => D::typenum_datetimeunit(N::NPY_TIMEDELTA, unit),
                }
            }
            TypeDescriptor::Array(arr) => D::array(&arr),
            TypeDescriptor::Record(rec) => D::record(&rec),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum TypenumExtra {
    Flexible(usize),
    Endian(Endian),
    DatetimeUnit(DatetimeUnit),
}

#[derive(Debug, Clone)]
pub enum DtypeArgs<'a> {
    Typenum(NPY_TYPES, Option<TypenumExtra>),
    Array(&'a ArrayDescriptor<Scalar>),
    Record(&'a RecordDescriptor<Scalar>),
}

impl<'a> DtypeArgs<'a> {
    pub fn typenum(typenum: NPY_TYPES) -> Self {
        Self::Typenum(typenum, None)
    }

    pub fn typenum_endian(typenum: NPY_TYPES, endian: Endian) -> Self {
        Self::Typenum(typenum, Some(TypenumExtra::Endian(endian)))
    }

    pub fn typenum_flexible(typenum: NPY_TYPES, elsize: usize) -> Self {
        Self::Typenum(typenum, Some(TypenumExtra::Flexible(elsize)))
    }

    pub fn typenum_datetimeunit(typenum: NPY_TYPES, unit: DatetimeUnit) -> Self {
        Self::Typenum(typenum, Some(TypenumExtra::DatetimeUnit(unit)))
    }

    pub fn array(arr: &'a ArrayDescriptor<Scalar>) -> Self {
        Self::Array(arr)
    }

    pub fn record(rec: &'a RecordDescriptor<Scalar>) -> Self {
        Self::Record(rec)
    }

    pub unsafe fn into_dtype(self, py: Python) -> PyResult<&PyArrayDescr> {
        let dtype = match self {
            Self::Typenum(typenum, extra) => {
                let mut dtype = PY_ARRAY_API.PyArray_DescrFromType(typenum as _);
                // TODO: check if null?
                match extra {
                    Some(TypenumExtra::Flexible(elsize)) => {
                        // TODO: checked cast? (hypothetical c_int overflow)
                        (*dtype).elsize = elsize as _;
                    }
                    Some(TypenumExtra::Endian(endian)) if endian.is_swapped() => {
                        // we bypass PyArray_DescrNewByteorder since it has lots of extra logic
                        // that we don't need (e.g. recursively setting endianness for subarrays
                        // and record type fields)
                        let dtype_swapped = PY_ARRAY_API.PyArray_DescrNew(dtype);
                        // TODO: check if null?
                        (*dtype_swapped).byteorder = NPY_BYTEORDER_CHAR::NPY_OPPBYTE as _;
                        Py_DECREF(dtype as _);
                        dtype = dtype_swapped;
                    }
                    Some(TypenumExtra::DatetimeUnit(unit)) => {
                        // an equivalent of create_datetime_dtype()
                        let meta =
                            PyArray_DatetimeMetaData { base: unit.into_npy_datetimeunit(), num: 1 };
                        let dtype_meta = (*dtype).c_metadata as *mut PyArray_DatetimeDTypeMetaData;
                        (*dtype_meta).meta = meta;
                    }
                    _ => {}
                }
                dtype
            }
            _ => todo!(),
        };
        Ok(py.from_owned_ptr(dtype as _))
    }
}
