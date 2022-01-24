#[macro_use]
pub mod macros;

pub mod datetime;
pub mod dtype;
pub mod element;

pub use datetime::{units, Datetime64, DatetimeUnit, Timedelta64};
pub use dtype::{dtype, dtype_from_type_descriptor};
pub use element::{ArrayElement, Scalar};

pub use pyo3_type_desc::Element;
