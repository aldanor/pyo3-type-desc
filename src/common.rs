use std::fmt::{self, Debug, Display};
use std::hash::{Hash, Hasher};
use std::ops::{Deref, Not};

#[derive(Clone)]
pub enum BoxCow<'a, T: ?Sized> {
    Borrowed(&'a T),
    Owned(Box<T>),
}

impl<'a, T: ?Sized> Deref for BoxCow<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Borrowed(borrowed) => borrowed,
            Self::Owned(owned) => &*owned,
        }
    }
}

impl<'a, T: ?Sized> AsRef<T> for BoxCow<'a, T> {
    fn as_ref(&self) -> &T {
        match self {
            Self::Borrowed(borrowed) => borrowed,
            Self::Owned(owned) => &*owned,
        }
    }
}

impl<'a, T: Clone + ?Sized> BoxCow<'a, T> {
    pub fn into_owned(self) -> Box<T> {
        match self {
            Self::Borrowed(borrowed) => Box::new(T::clone(borrowed)),
            Self::Owned(owned) => owned,
        }
    }
}

impl<'a, T: PartialEq + ?Sized> PartialEq for BoxCow<'a, T> {
    fn eq(&self, other: &Self) -> bool {
        self.as_ref().eq(other.as_ref())
    }
}

impl<'a, T: PartialEq + Eq + ?Sized> Eq for BoxCow<'a, T> {}

impl<'a, T: ?Sized> From<&'a T> for BoxCow<'a, T> {
    fn from(borrowed: &'a T) -> Self {
        Self::Borrowed(borrowed)
    }
}

impl<'a, T: ?Sized> From<Box<T>> for BoxCow<'a, T> {
    fn from(owned: Box<T>) -> Self {
        Self::Owned(owned)
    }
}

impl<'a, T: Debug + ?Sized> Debug for BoxCow<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Borrowed(borrowed) => write!(f, "{:?}", borrowed),
            Self::Owned(owned) => write!(f, "{:?}", owned),
        }
    }
}

impl<'a, T: Display + ?Sized> Display for BoxCow<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Borrowed(borrowed) => write!(f, "{}", borrowed),
            Self::Owned(owned) => write!(f, "{}", owned),
        }
    }
}

impl<'a, T: Hash + ?Sized> Hash for BoxCow<'a, T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Hash::hash(&**self, state)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Endian {
    Little,
    Big,
}

impl Endian {
    #[cfg(target_endian = "little")]
    pub const NATIVE: Self = Self::Little;
    #[cfg(target_endian = "big")]
    pub const NATIVE: Self = Self::Big;

    pub fn is_native(self) -> bool {
        self == Self::NATIVE
    }

    pub fn is_swapped(self) -> bool {
        !self.is_native()
    }
}

impl Default for Endian {
    fn default() -> Self {
        Self::NATIVE
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Signedness {
    Signed = 0,
    Unsigned = 1,
}

impl Signedness {
    pub fn of<T: Default + Not<Output = T> + PartialOrd>() -> Self {
        if T::default() > !T::default() {
            Self::Signed
        } else {
            Self::Unsigned
        }
    }

    pub fn is_signed(self) -> bool {
        self == Self::Signed
    }

    pub fn is_unsigned(self) -> bool {
        self == Self::Unsigned
    }
}
