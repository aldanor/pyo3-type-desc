use std::fmt::{self, Display};
use std::marker::PhantomData;

use numpy::npyffi::NPY_DATETIMEUNIT;

use pyo3_type_desc::{Element, TypeDescriptor};

use crate::Scalar;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(C)]
pub enum DatetimeUnit {
    Year = 0,
    Month = 1,
    Week = 2,
    // there is no '3'
    Day = 4,
    Hour = 5,
    Minute = 6,
    Second = 7,
    Millisecond = 8,
    Microsecond = 9,
    Nanosecond = 10,
    Picosecond = 11,
    Femtosecond = 12,
    Attosecond = 13,
    // technically, there's also 'generic' unit ('14') but it can only be used for timedeltas
}

impl DatetimeUnit {
    pub fn to_code(self) -> &'static str {
        match self {
            Self::Year => "Y",
            Self::Month => "M",
            Self::Week => "W",
            Self::Day => "D",
            Self::Hour => "h",
            Self::Minute => "m",
            Self::Second => "s",
            Self::Millisecond => "ms",
            Self::Microsecond => "us",
            Self::Nanosecond => "ns",
            Self::Picosecond => "ps",
            Self::Femtosecond => "fs",
            Self::Attosecond => "as",
        }
    }

    pub fn from_code(code: impl AsRef<[u8]>) -> Option<Self> {
        Some(match code.as_ref() {
            b"Y" => Self::Year,
            b"M" => Self::Month,
            b"W" => Self::Week,
            b"D" => Self::Day,
            b"h" => Self::Hour,
            b"m" => Self::Minute,
            b"s" => Self::Second,
            b"ms" => Self::Millisecond,
            b"us" => Self::Microsecond,
            b"ns" => Self::Nanosecond,
            b"ps" => Self::Picosecond,
            b"fs" => Self::Femtosecond,
            b"as" => Self::Attosecond,
            _ => return None,
        })
    }

    pub fn into_npy_datetimeunit(self) -> NPY_DATETIMEUNIT {
        type N = NPY_DATETIMEUNIT;
        match self {
            Self::Year => N::NPY_FR_Y,
            Self::Month => N::NPY_FR_M,
            Self::Week => N::NPY_FR_W,
            Self::Day => N::NPY_FR_D,
            Self::Hour => N::NPY_FR_h,
            Self::Minute => N::NPY_FR_m,
            Self::Second => N::NPY_FR_s,
            Self::Millisecond => N::NPY_FR_ms,
            Self::Microsecond => N::NPY_FR_us,
            Self::Nanosecond => N::NPY_FR_ns,
            Self::Picosecond => N::NPY_FR_ps,
            Self::Femtosecond => N::NPY_FR_fs,
            Self::Attosecond => N::NPY_FR_as,
        }
    }
}

impl Display for DatetimeUnit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(self.to_code())
    }
}

pub mod units {
    use std::fmt::Debug;
    use std::hash::Hash;

    use super::DatetimeUnit;

    pub trait Unit: Clone + Copy + PartialEq + Eq + PartialOrd + Ord + Hash + Debug + Send {
        const UNIT: DatetimeUnit;
    }

    pub trait LinearUnit: Unit {}

    macro_rules! define_unit {
        ($ty:ident) => {
            #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
            pub struct $ty;

            impl Unit for $ty {
                const UNIT: DatetimeUnit = DatetimeUnit::$ty;
            }
        };

        ($ty:ident: linear) => {
            define_unit!($ty);

            impl LinearUnit for $ty {}
        };
    }

    define_unit!(Year);
    define_unit!(Month);
    define_unit!(Week: linear);
    define_unit!(Day: linear);
    define_unit!(Hour: linear);
    define_unit!(Minute: linear);
    define_unit!(Second: linear);
    define_unit!(Millisecond: linear);
    define_unit!(Microsecond: linear);
    define_unit!(Nanosecond: linear);
    define_unit!(Picosecond: linear);
    define_unit!(Femtosecond: linear);
    define_unit!(Attosecond: linear);
}

use self::units::Unit;

#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Datetime64<U: Unit>(i64, PhantomData<U>);

#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Timedelta64<U: Unit>(i64, PhantomData<U>);

macro_rules! impl_datetime {
    ($ty:ident, $scalar:ident) => {
        impl<U: Unit> $ty<U> {
            const UNIT: DatetimeUnit = U::UNIT;

            pub fn new(value: i64) -> Self {
                Self(value, PhantomData::default())
            }

            pub fn value(self) -> i64 {
                self.0
            }
        }

        impl<U: Unit> From<i64> for $ty<U> {
            fn from(value: i64) -> Self {
                Self::new(value)
            }
        }

        impl<U: Unit> From<$ty<U>> for i64 {
            fn from(value: $ty<U>) -> Self {
                value.value()
            }
        }

        unsafe impl<U: Unit> Element<Scalar> for $ty<U> {
            fn type_descriptor() -> TypeDescriptor<Scalar> {
                debug_assert_eq!(std::mem::size_of::<$ty<U>>(), 8);
                debug_assert_eq!(std::mem::align_of::<$ty<U>>(), 8);
                Scalar::$scalar(Self::UNIT).into()
            }
        }
    };
}

impl_datetime!(Datetime64, Datetime);
impl_datetime!(Timedelta64, Timedelta);

// TODO: arithmetics / reprs / conversion to/from stdlib type and time types
