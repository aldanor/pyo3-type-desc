#![allow(non_camel_case_types)]

pub use self::array::*;
pub use self::objects::*;
pub use self::types::*;

pub mod array {
    // these are under `#if NPY_USE_PYMEM == 1` which seems to be always defined as 1
    pub use pyo3::ffi::PyMem_RawFree as PyArray_free;
    pub use pyo3::ffi::PyMem_RawMalloc as PyArray_malloc;
    pub use pyo3::ffi::PyMem_RawRealloc as PyArray_realloc;
}

pub mod types {
    #[repr(u8)]
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub enum NPY_BYTEORDER_CHAR {
        NPY_LITTLE = b'<',
        NPY_BIG = b'>',
        NPY_NATIVE = b'=',
        NPY_SWAP = b's',
        NPY_IGNORE = b'|',
    }

    impl NPY_BYTEORDER_CHAR {
        #[cfg(target_endian = "little")]
        pub const NPY_NATBYTE: Self = Self::NPY_LITTLE;
        #[cfg(target_endian = "little")]
        pub const NPY_OPPBYTE: Self = Self::NPY_BIG;

        #[cfg(target_endian = "big")]
        pub const NPY_NATBYTE: Self = Self::NPY_BIG;
        #[cfg(target_endian = "big")]
        pub const NPY_OPPBYTE: Self = Self::NPY_LITTLE;
    }
}

pub mod objects {
    use std::os::raw::c_int;

    use numpy::npyffi::{NpyAuxData, NPY_DATETIMEUNIT};

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct PyArray_DatetimeMetaData {
        pub base: NPY_DATETIMEUNIT,
        pub num: c_int,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct PyArray_DatetimeDTypeMetaData {
        pub base: NpyAuxData,
        pub meta: PyArray_DatetimeMetaData,
    }
}
