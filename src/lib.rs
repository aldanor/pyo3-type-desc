pub mod common;
pub mod desc;
pub mod element;
pub mod parse;
pub mod scalar;

pub use {
    common::{BoxCow, Endian, Signedness},
    desc::{ArrayDescriptor, FieldDescriptor, RecordDescriptor, ScalarDescriptor, TypeDescriptor},
    element::Element,
    scalar::{ComplexSize, FloatSize, IntegerSize, Scalar},
};
