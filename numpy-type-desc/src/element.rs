use pyo3_type_desc::{Scalar as BaseScalar, ScalarDescriptor};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Scalar {
    Base(BaseScalar),
    Datetime,
    Timedelta,
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
