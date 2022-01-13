use crate::common::{Endian, Signedness};
use crate::desc::TypeDescriptor;
use crate::scalar::{FloatSize, IntegerSize, Scalar};

#[cfg(feature = "complex")]
use crate::scalar::ComplexSize;

pub unsafe trait Element: Copy {
    const TYPE: TypeDescriptor<Scalar>;
}

type _S = Scalar;
type _TD = TypeDescriptor<_S>;

macro_rules! impl_element {
    ($ty:ty, $expr:expr) => {
        unsafe impl Element for $ty {
            const TYPE: _TD = _TD::Scalar($expr);
        }
    };
    ($ty:ty => $variant:ident) => {
        impl_element!($ty, _S::$variant);
    };
    (char: $ty:ty => $sign:ident) => {
        impl_element!($ty, _S::Char(Signedness::$sign));
    };
    (int: $ty:ty => $size:ident, $sign:ident) => {
        impl_element!($ty, _S::Integer(IntegerSize::$size, Signedness::$sign, Endian::NATIVE));
    };
    (float: $ty:ty => $size:ident) => {
        impl_element!($ty, _S::Float(FloatSize::$size, Endian::NATIVE));
    };
    (complex: $ty:ty => $size:ident) => {
        impl_element!($ty, _S::Complex(ComplexSize::$size, Endian::NATIVE));
    };
}

impl_element!(bool => Bool);

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
