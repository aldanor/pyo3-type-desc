use std::mem;

use memoffset::offset_of_tuple;
use pyo3::PyObject;

use crate::common::{Endian, Signedness};
use crate::desc::{FieldDescriptor, RecordDescriptor, ScalarDescriptor, TypeDescriptor};
use crate::scalar::{FloatSize, IntegerSize, Scalar};

#[cfg(feature = "complex")]
use crate::scalar::ComplexSize;

pub unsafe trait Element<S: ScalarDescriptor>: Clone + Send {
    fn type_descriptor() -> TypeDescriptor<S>;
}

type _S = Scalar;

macro_rules! impl_element {
    ($ty:ty => $expr:expr) => {
        unsafe impl<S: ScalarDescriptor + From<Scalar>> Element<S> for $ty {
            #[inline]
            fn type_descriptor() -> TypeDescriptor<S> {
                let desc: TypeDescriptor<S> = $expr;
                debug_assert_eq!(std::mem::size_of::<$ty>(), desc.itemsize());
                $expr
            }
        }
    };
    (@s: $ty:ty, $expr:expr) => {
        impl_element!($ty => TypeDescriptor::Scalar($expr.into()));
    };
    (scalar: $ty:ty => $variant:ident) => {
        impl_element!(@s: $ty, _S::$variant);
    };
    (char: $ty:ty => $sign:ident) => {
        impl_element!(@s: $ty, _S::Char(Signedness::$sign));
    };
    (int: $ty:ty => $size:ident, $sign:ident) => {
        impl_element!(@s: $ty, _S::Integer(IntegerSize::$size, Signedness::$sign, Endian::NATIVE));
    };
    (float: $ty:ty => $size:ident) => {
        impl_element!(@s: $ty, _S::Float(FloatSize::$size, Endian::NATIVE));
    };
    (complex: $ty:ty => $size:ident) => {
        impl_element!(@s: $ty, _S::Complex(ComplexSize::$size, Endian::NATIVE));
    };
}

impl_element!(scalar: bool => Bool);

impl_element!(char: i8 => Signed);
impl_element!(char: u8 => Unsigned);

impl_element!(int: i16 => Two, Signed);
impl_element!(int: i32 => Four, Signed);
impl_element!(int: i64 => Eight, Signed);
impl_element!(int: u16 => Two, Unsigned);
impl_element!(int: u32 => Four, Unsigned);
impl_element!(int: u64 => Eight, Unsigned);

impl_element!(float: f32 => Four);
impl_element!(float: f64 => Eight);

#[cfg(target_pointer_width = "32")]
impl_element!(int: usize => Four, Unsigned);
#[cfg(target_pointer_width = "32")]
impl_element!(int: isize => Four, Signed);

#[cfg(target_pointer_width = "64")]
impl_element!(int: usize => Eight, Unsigned);
#[cfg(target_pointer_width = "64")]
impl_element!(int: isize => Eight, Signed);

#[cfg(feature = "complex")]
impl_element!(complex: num_complex::Complex32 => Four);
#[cfg(feature = "complex")]
impl_element!(complex: num_complex::Complex64 => Eight);

impl_element!(PyObject => TypeDescriptor::Object);

macro_rules! impl_element_tuple {
    ($($field:tt, $ty:ident);+) => {
        unsafe impl<S: ScalarDescriptor, $($ty),+> Element<S> for ($($ty,)+)
        where
            $($ty: Element<S>,)+
        {
            #[inline]
            fn type_descriptor() -> TypeDescriptor<S> {
                TypeDescriptor::Record(RecordDescriptor {
                    fields: vec![$(
                        FieldDescriptor {
                            desc: <$ty>::type_descriptor(),
                            name: None,
                            offset: offset_of_tuple!(Self, $field),
                        }
                    ),+].into(),
                    itemsize: mem::size_of::<Self>(),
                    alignment: Some(mem::align_of::<Self>()),
                })
            }
        }
    };
}

impl_element_tuple!(0, A);
impl_element_tuple!(0, A; 1, B);
impl_element_tuple!(0, A; 1, B; 2, C);
impl_element_tuple!(0, A; 1, B; 2, C; 3, D);
impl_element_tuple!(0, A; 1, B; 2, C; 3, D; 4, E);
impl_element_tuple!(0, A; 1, B; 2, C; 3, D; 4, E; 5, F);
impl_element_tuple!(0, A; 1, B; 2, C; 3, D; 4, E; 5, F; 6, G);
impl_element_tuple!(0, A; 1, B; 2, C; 3, D; 4, E; 5, F; 6, G; 7, H);
impl_element_tuple!(0, A; 1, B; 2, C; 3, D; 4, E; 5, F; 6, G; 7, H; 8, I);
impl_element_tuple!(0, A; 1, B; 2, C; 3, D; 4, E; 5, F; 6, G; 7, H; 8, I; 9, J);
impl_element_tuple!(0, A; 1, B; 2, C; 3, D; 4, E; 5, F; 6, G; 7, H; 8, I; 9, J; 10, K);
impl_element_tuple!(0, A; 1, B; 2, C; 3, D; 4, E; 5, F; 6, G; 7, H; 8, I; 9, J; 10, K; 11, L);

#[rustversion::since(1.51)]
unsafe impl<S: ScalarDescriptor, T: Element<S>, const N: usize> Element<S> for [T; N] {
    fn type_descriptor() -> TypeDescriptor<S> {
        // just a note: array size may be zero
        use crate::ArrayDescriptor;
        TypeDescriptor::Array(match T::type_descriptor() {
            TypeDescriptor::Array(ArrayDescriptor { desc, shape }) => {
                // TODO: do we want to always collapse shapes? should this be preventable or not?
                let mut newshape = Vec::with_capacity(shape.len() + 1);
                newshape.push(N);
                newshape.extend(shape.iter().copied());
                ArrayDescriptor { desc, shape: newshape.into() }
            }
            desc => ArrayDescriptor { desc: desc.into(), shape: [N].as_ref().into() },
        })
    }
}
