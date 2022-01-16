#![allow(non_camel_case_types)]

pub use self::array::*;
pub use self::flags::*;
pub use self::objects::*;
pub use self::types::*;

pub mod array {
    // these are under `#if NPY_USE_PYMEM == 1` which seems to be always defined as 1
    pub use pyo3::ffi::PyMem_RawFree as PyArray_free;
    pub use pyo3::ffi::PyMem_RawMalloc as PyArray_malloc;
    pub use pyo3::ffi::PyMem_RawRealloc as PyArray_realloc;
}

pub mod flags {
    use numpy::npyffi::npy_char;

    pub const NPY_ITEM_REFCOUNT: npy_char = 0x01;
    pub const NPY_ITEM_HASOBJECT: npy_char = 0x01;
    pub const NPY_LIST_PICKLE: npy_char = 0x02;
    pub const NPY_ITEM_IS_POINTER: npy_char = 0x04;
    pub const NPY_NEEDS_INIT: npy_char = 0x08;
    pub const NPY_NEEDS_PYAPI: npy_char = 0x10;
    pub const NPY_USE_GETITEM: npy_char = 0x20;
    pub const NPY_USE_SETITEM: npy_char = 0x40;
    pub const NPY_ALIGNED_STRUCT: npy_char = -128; // 0x80;

    pub const NPY_FROM_FIELDS: npy_char =
        NPY_NEEDS_INIT | NPY_LIST_PICKLE | NPY_ITEM_REFCOUNT | NPY_NEEDS_PYAPI;

    pub const NPY_OBJECT_DTYPE_FLAGS: npy_char = NPY_LIST_PICKLE
        | NPY_USE_GETITEM
        | NPY_ITEM_IS_POINTER
        | NPY_ITEM_REFCOUNT
        | NPY_NEEDS_INIT
        | NPY_NEEDS_PYAPI;
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
