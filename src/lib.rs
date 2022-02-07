#![allow(clippy::missing_safety_doc)]

pub mod common;
pub mod desc;
pub mod element;
pub mod parse;
pub mod scalar;

pub use {
    common::{BoxCow, Endian, Signedness},
    desc::{ArrayDescriptor, FieldDescriptor, RecordDescriptor, ScalarDescriptor, TypeDescriptor},
    element::{BufferElement, Element},
    scalar::{ComplexSize, FloatSize, IntegerSize, Scalar},
};
