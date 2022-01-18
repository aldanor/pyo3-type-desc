#[macro_use]
pub mod macros;

pub mod datetime;
pub mod dtype;
pub mod element;
pub mod npyffi;

pub use datetime::DatetimeUnit;
pub use dtype::{dtype, dtype_from_type_descriptor};
pub use element::Scalar;

pub use pyo3_type_desc::Element;
