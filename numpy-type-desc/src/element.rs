use std::fmt::{self, Debug};

use pyo3_type_desc::{Element, Scalar as BaseScalar, ScalarDescriptor};

use crate::datetime::DatetimeUnit;

pub unsafe trait ArrayElement: Element<Scalar> {}
unsafe impl<T: Element<Scalar>> ArrayElement for T {}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub enum Scalar {
    Base(BaseScalar),
    Datetime(DatetimeUnit),
    Timedelta(DatetimeUnit),
}

impl From<BaseScalar> for Scalar {
    fn from(scalar: BaseScalar) -> Self {
        Self::Base(scalar)
    }
}

impl ScalarDescriptor for Scalar {
    fn itemsize(&self) -> usize {
        match *self {
            Self::Base(desc) => desc.itemsize(),
            _ => 8,
        }
    }

    fn alignment(&self) -> usize {
        match *self {
            Self::Base(desc) => desc.alignment(),
            _ => 8,
        }
    }
}

impl Debug for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Base(scalar) => write!(f, "{:?}", scalar),
            Self::Datetime(units) => write!(f, "datetime64[{}]", units),
            Self::Timedelta(units) => write!(f, "timedelta64[{}]", units),
        }
    }
}
