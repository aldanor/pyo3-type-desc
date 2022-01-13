use std::fmt::{self, Debug, Display};

use crate::common::{Endian, Signedness};
use crate::desc::ScalarDescriptor;

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(usize)]
pub enum IntegerSize {
    Two = 2,
    Four = 4,
    Eight = 8,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(usize)]
pub enum FloatSize {
    Two = 2,
    Four = 4,
    Eight = 8,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(usize)]
pub enum ComplexSize {
    Four = 4,
    Eight = 8,
}

macro_rules! impl_fmt_size {
    ($ty:ty: $trait:ty) => {
        impl $trait for $ty {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "{}", *self as usize)
            }
        }
    };
    ($ty:ty) => {
        impl_fmt_size!($ty: Debug);
        impl_fmt_size!($ty: Display);
    };
}

impl_fmt_size!(IntegerSize);
impl_fmt_size!(FloatSize);
impl_fmt_size!(ComplexSize);

// Types explicitly listed as unsupported in numpy:
// - bitfields (t)
// - pointers (&)
// - UCS-2 strings (u)
// - functions (X)
// Types listed in struct but implicitly unsupported in numpy:
// - Pascal strings (p)
// - void* pointers (P)
// - pointer-size ints (n, N)
// Other types we're not going to support:
// - long double (g) - see e.g. https://github.com/rust-lang/rust-bindgen/issues/1549
// - long complex (Zg) - same as above, doesn't make sense without compiler support
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferScalar {
    /// Boolean type
    ///
    /// | code     | type     |
    /// |----------|----------|
    /// | **`?`**  | [`bool`] |
    ///
    /// Corresponds to the `_Bool` type defined by C99. Always represented by one byte.
    Bool,
    /// Single-byte integer
    ///
    /// | code     | type   |
    /// |----------|--------|
    /// | **`b`**  | [`i8`] |
    /// | **`B`**  | [`u8`] |
    Char(Signedness),
    /// Multi-byte integer
    ///
    /// | code     | standard   | native                                         |
    /// |----------|------------|------------------------------------------------|
    /// | **`h`**  | [`i16`]    | [`c_short`](std::os::raw::c_short) = [`i16`]   |
    /// | **`H`**  | [`u16`]    | [`c_ushort`](std::os::raw::c_ushort) = [`u16`] |
    /// | **`i`**  | [`i32`]    | [`c_int`](std::os::raw::c_int)                 |
    /// | **`I`**  | [`u32`]    | [`c_uint`](std::os::raw::c_uint)               |
    /// | **`l`**  | [`i32`]    | [`c_long`](std::os::raw::c_long)               |
    /// | **`L`**  | [`u32`]    | [`c_ulong`](std::os::raw::c_ulong)             |
    /// | **`q`**  | [`i64`]    | [`c_longlong`](std::os::raw::c_longlong)       |
    /// | **`Q`**  | [`u64`]    | [`c_ulonglong`](std::os::raw::c_ulonglong)     |
    ///
    /// In "standard mode", all integer types are fixed, while in "native mode",
    /// they are platform-dependent (except for 2-byte integers). Native mode
    /// is enabled by default and can be set via `@` and `^` modifiers in the
    /// format string.
    Integer(IntegerSize, Signedness, Endian),
    /// Floating-point number
    ///
    /// | code     | type    |
    /// |----------|---------|
    /// | **`e`**  | `f16`   |
    /// | **`f`**  | [`f32`] |
    /// | **`d`**  | [`f64`] |
    ///
    /// Note: long double (`g`) is not listed due to the lack of compiler support.
    Float(FloatSize, Endian),
    /// | code      | type           |
    /// |-----------|----------------|
    /// | **`Zf`**  | `Complex<f32>` |
    /// | **`Zd`**  | `Complex<f64>` |
    ///
    /// Note: long complex (`Zg`) is not listed due to the lack of compiler support.
    Complex(ComplexSize, Endian),
    /// Fixed-length byte string
    ///
    /// | code              | type      |
    /// |--------------- ---|-----------|
    /// | **`(?<N>\d*)s`**  | `[u8; N]` |
    Bytes(usize), // TODO: NonZeroUsize? numpy translates 0s -> |S0 -> 1s (!)
    /// Fixed-length wide (UCS-4) byte string
    ///
    /// | code              | type           |
    /// |------------- -----|----------------|
    /// | **`(?<N>\d*)w`**  | `[[u8; 4]; N]` |
    Ucs4(usize), // TODO: NonZeroUsize? numpy translates 0w -> |U0 -> 1w (!)
}

impl BufferScalar {
    /// Returns true if the type is a byte array or a UCS-4 string.
    pub fn is_array_like(&self) -> bool {
        matches!(self, Self::Bytes(_) | Self::Ucs4(_))
    }
}

impl ScalarDescriptor for BufferScalar {
    fn itemsize(&self) -> usize {
        match *self {
            Self::Bool | Self::Char(_) => 1,
            Self::Integer(size, ..) => size as usize,
            Self::Float(size, ..) => size as usize,
            Self::Complex(size, ..) => 2 * size as usize,
            Self::Bytes(size) => size,
            Self::Ucs4(size) => size * 4,
        }
    }

    fn alignment(&self) -> usize {
        match *self {
            Self::Bytes(_) => 1,
            Self::Ucs4(_) => 4,
            Self::Complex(size, ..) => size as usize,
            _ => self.itemsize(),
        }
    }
}

impl Debug for BufferScalar {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let int_char = |s: Signedness| if s.is_signed() { 'i' } else { 'u' };
        let endian_char = |e: Endian| if e == Endian::Little { '<' } else { '>' };
        let repr = match *self {
            Self::Bool => "bool".into(),
            Self::Char(sign) => format!("{}8", int_char(sign)),
            Self::Integer(size, sign, endian) => {
                format!("{}{}{}", endian_char(endian), int_char(sign), (size as usize) * 8)
            }
            Self::Float(size, endian) => {
                format!("{}f{}", endian_char(endian), (size as usize) * 8)
            }
            Self::Complex(size, endian) => {
                format!("{}c{}", endian_char(endian), (size as usize) * 8)
            }
            Self::Bytes(size) => format!("S{}", size),
            Self::Ucs4(size) => format!("U{}", size),
        };
        write!(f, "{}", repr)
    }
}
