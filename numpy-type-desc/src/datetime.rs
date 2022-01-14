use std::fmt::{self, Display};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum DatetimeUnit {
    Date(DateUnit),
    Time(TimeUnit),
}

impl DatetimeUnit {
    pub fn to_code(self) -> &'static str {
        match self {
            Self::Date(date) => date.to_code(),
            Self::Time(time) => time.to_code(),
        }
    }

    pub fn from_code(code: impl AsRef<[u8]>) -> Option<Self> {
        let code = code.as_ref();
        DateUnit::from_code(code)
            .map(Self::Date)
            .or_else(|| TimeUnit::from_code(code).map(Self::Time))
    }
}

impl Display for DatetimeUnit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(self.to_code())
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum DateUnit {
    Year,
    Month,
    Week,
    Day,
}

impl DateUnit {
    pub fn to_code(self) -> &'static str {
        match self {
            Self::Year => "Y",
            Self::Month => "M",
            Self::Week => "W",
            Self::Day => "D",
        }
    }

    pub fn from_code(code: impl AsRef<[u8]>) -> Option<Self> {
        Some(match code.as_ref() {
            b"Y" => Self::Year,
            b"M" => Self::Month,
            b"W" => Self::Week,
            b"D" => Self::Day,
            _ => return None,
        })
    }
}

impl Display for DateUnit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(self.to_code())
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum TimeUnit {
    Hour,
    Minute,
    Second,
    Millisecond,
    Microsecond,
    Nanosecond,
    Picosecond,
    Femtosecond,
    Attosecond,
}

impl TimeUnit {
    pub fn to_code(self) -> &'static str {
        match self {
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
}

impl Display for TimeUnit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(self.to_code())
    }
}
