use std::fmt::{self, Display};

use numpy::npyffi::NPY_DATETIMEUNIT;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(C)]
pub enum DatetimeUnit {
    Year = 0,
    Month = 1,
    Week = 2,
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
    // TODO: technically, there's also 'generic' unit (14) which can be only used for timedeltas
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
