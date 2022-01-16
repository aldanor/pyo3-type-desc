use std::borrow::Cow;
use std::fmt::{self, Debug};
use std::mem;

use crate::common::BoxCow;

pub trait ScalarDescriptor: 'static + Clone {
    fn itemsize(&self) -> usize;
    fn alignment(&self) -> usize;
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct FieldDescriptor<T: ScalarDescriptor> {
    pub desc: TypeDescriptor<T>,
    pub name: Option<Cow<'static, str>>,
    pub offset: usize,
}

impl<T: ScalarDescriptor> FieldDescriptor<T> {
    pub fn new<N>(desc: TypeDescriptor<T>, name: Option<N>, offset: usize) -> Self
    where
        N: Into<Cow<'static, str>>,
    {
        Self { desc, name: name.map(Into::into), offset }
    }

    pub fn has_object(&self) -> bool {
        self.desc.has_object()
    }

    pub fn map<U: ScalarDescriptor, F: Copy + Fn(&T) -> U>(&self, func: F) -> FieldDescriptor<U> {
        FieldDescriptor { desc: self.desc.map(func), name: self.name.clone(), offset: self.offset }
    }
}

impl<T: ScalarDescriptor + Debug> Debug for FieldDescriptor<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(name) = self.name.as_ref() {
            write!(f, "{:?}:", name)?;
        }
        write!(f, "{} => {:?}", self.offset, self.desc)
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct RecordDescriptor<T: ScalarDescriptor> {
    pub fields: Cow<'static, [FieldDescriptor<T>]>,
    pub itemsize: usize, // TODO: NonZeroUsize? do we allow zero-size struct types?
    pub alignment: Option<usize>,
}

impl<T: ScalarDescriptor> RecordDescriptor<T> {
    // TODO: add validation? fields don't overlap and don't go past itemsize, etc
    pub fn new<F>(fields: F, itemsize: usize, alignment: Option<usize>) -> Self
    where
        F: Into<Cow<'static, [FieldDescriptor<T>]>>,
    {
        Self { fields: fields.into(), itemsize, alignment }
    }

    pub fn has_object(&self) -> bool {
        self.fields.iter().any(FieldDescriptor::has_object)
    }

    pub fn map<U: ScalarDescriptor, F: Copy + Fn(&T) -> U>(&self, func: F) -> RecordDescriptor<U> {
        RecordDescriptor {
            fields: Cow::Owned(self.fields.iter().map(|field| field.map(func)).collect()),
            itemsize: self.itemsize,
            alignment: self.alignment,
        }
    }

    pub fn is_aligned(&self) -> bool {
        // TODO: zero-field structs? zero-sized structs? any other corner cases?
        self.alignment.map_or(false, |align| {
            let itemsize_ok = self.itemsize % align == 0;
            // TODO: in alignment_ok: equality or divisibility?
            let alignment_ok =
                align == self.fields.iter().map(|field| field.desc.alignment()).max().unwrap_or(1);
            let offsets_ok =
                self.fields.iter().all(|field| field.offset % field.desc.alignment() == 0);
            itemsize_ok && alignment_ok && offsets_ok
        })
    }

    pub fn fill_names(&self) -> impl Iterator<Item = Cow<'static, str>> + '_ {
        let (mut i, mut n) = (0, String::from("f0"));
        self.fields.iter().map(move |field| match field.name.as_ref() {
            Some(name) => name.clone(),
            _ => {
                while self
                    .fields
                    .iter()
                    .any(|field| field.name.as_ref().map_or(false, |name| name.as_ref() == &n))
                {
                    i += 1;
                    n = format!("f{}", i);
                }
                let name = mem::take(&mut n);
                i += 1;
                n = format!("f{}", i);
                Cow::Owned(name)
            }
        })
    }
}

impl<T: ScalarDescriptor + Debug> Debug for RecordDescriptor<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{{")?;
        for (i, field) in self.fields.iter().enumerate() {
            if i != 0 {
                write!(f, ", ")?;
            }
            write!(f, "{:?}", field)?;
        }
        if !self.fields.is_empty() {
            write!(f, " ")?;
        }
        write!(f, "[{}", self.itemsize)?;
        if let Some(align) = self.alignment {
            write!(f, ", {}", align)?;
        }
        write!(f, "]}}")
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ArrayDescriptor<T: ScalarDescriptor> {
    pub desc: BoxCow<'static, TypeDescriptor<T>>,
    pub shape: Cow<'static, [usize]>, // TODO: NonZeroUsize? do we allow zero-sized subarrays?
}

impl<T: ScalarDescriptor> ArrayDescriptor<T> {
    pub fn new<D, S>(desc: D, shape: S) -> Self
    where
        D: Into<BoxCow<'static, TypeDescriptor<T>>>,
        S: Into<Cow<'static, [usize]>>,
    {
        Self { desc: desc.into(), shape: shape.into() }
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn itemsize(&self) -> usize {
        self.size() * self.desc.itemsize()
    }

    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn has_object(&self) -> bool {
        self.desc.has_object()
    }

    pub fn map<U: ScalarDescriptor, F: Copy + Fn(&T) -> U>(&self, func: F) -> ArrayDescriptor<U> {
        ArrayDescriptor { desc: Box::new(self.desc.map(func)).into(), shape: self.shape.to_owned() }
    }
}

impl<T: ScalarDescriptor + Debug> Debug for ArrayDescriptor<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[(")?;
        for (i, dim) in self.shape.iter().enumerate() {
            if i != 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", dim)?;
        }
        write!(f, "); {:?}]", self.desc)
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub enum TypeDescriptor<T: ScalarDescriptor> {
    Object,
    Scalar(T),
    Array(ArrayDescriptor<T>),
    Record(RecordDescriptor<T>),
}

impl<T: ScalarDescriptor> TypeDescriptor<T> {
    pub fn object() -> Self {
        Self::Object
    }

    pub fn scalar(desc: T) -> Self {
        Self::Scalar(desc)
    }

    pub fn array<D, S>(desc: D, shape: S) -> Self
    where
        D: Into<BoxCow<'static, TypeDescriptor<T>>>,
        S: Into<Cow<'static, [usize]>>,
    {
        Self::Array(ArrayDescriptor::new(desc, shape))
    }

    pub fn record<F>(fields: F, itemsize: usize, alignment: Option<usize>) -> Self
    where
        F: Into<Cow<'static, [FieldDescriptor<T>]>>,
    {
        Self::Record(RecordDescriptor::new(fields, itemsize, alignment))
    }

    pub fn itemsize(&self) -> usize {
        match self {
            Self::Object => mem::size_of::<usize>(), // TODO: PyObject*? is this correct?
            Self::Scalar(desc) => desc.itemsize(),
            Self::Array(arr) => arr.itemsize(),
            Self::Record(rec) => rec.itemsize,
        }
    }

    pub fn alignment(&self) -> usize {
        match self {
            Self::Object => mem::size_of::<usize>(), // TODO: PyObject*? is this correct?
            Self::Scalar(desc) => desc.alignment(),
            Self::Array(arr) => arr.desc.alignment(),
            Self::Record(rec) => rec.alignment.unwrap_or(1),
        }
    }

    pub fn has_object(&self) -> bool {
        match self {
            Self::Object => true,
            Self::Scalar(_) => false,
            Self::Array(arr) => arr.has_object(),
            Self::Record(rec) => rec.has_object(),
        }
    }

    pub fn map<U: ScalarDescriptor, F: Copy + Fn(&T) -> U>(&self, func: F) -> TypeDescriptor<U> {
        match self {
            Self::Object => TypeDescriptor::Object,
            Self::Scalar(val) => TypeDescriptor::Scalar(func(val)),
            Self::Array(arr) => TypeDescriptor::Array(arr.map(func)),
            Self::Record(rec) => TypeDescriptor::Record(rec.map(func)),
        }
    }
}

impl<T: ScalarDescriptor + Debug> Debug for TypeDescriptor<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Object => write!(f, "object"),
            Self::Scalar(desc) => write!(f, "{:?}", desc),
            Self::Array(arr) => write!(f, "{:?}", arr),
            Self::Record(rec) => write!(f, "{:?}", rec),
        }
    }
}
