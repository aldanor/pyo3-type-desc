use std::mem;
use std::ops::{Deref, Not};
use std::os::raw::{c_int, c_long, c_longlong, c_short, c_uint, c_ulong, c_ulonglong, c_ushort};

use crate::common::{Endian, Signedness};
use crate::desc::{FieldDescriptor, ScalarDescriptor, TypeDescriptor};
use crate::scalar::{ComplexSize, FloatSize, IntegerSize, Scalar};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ByteStream<'a>(&'a [u8]);

impl<'a> Deref for ByteStream<'a> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl<'a, T: AsRef<[u8]> + ?Sized> From<&'a T> for ByteStream<'a> {
    fn from(s: &'a T) -> Self {
        Self(s.as_ref())
    }
}

impl<'a> ByteStream<'a> {
    pub fn new<T: Into<Self>>(s: T) -> Self {
        s.into()
    }

    pub fn advance(&mut self, n: usize) {
        self.0 = &self.0[n..];
    }

    pub fn peek(&self) -> Option<u8> {
        self.0.first().copied()
    }

    pub fn parse_usize(&mut self) -> Option<usize> {
        let (ch, mut s) = self.0.split_first()?;
        if !ch.is_ascii_digit() {
            return None;
        }
        let mut v = usize::from(*ch - b'0');
        while let Some((ch, tail)) = s.split_first() {
            if !ch.is_ascii_digit() {
                break;
            }
            v = v.checked_mul(10)?.checked_add(usize::from(*ch - b'0'))?;
            s = tail;
        }
        self.0 = s;
        Some(v)
    }

    pub fn consume_first(&mut self) -> Option<u8> {
        let (ch, tail) = self.0.split_first()?;
        self.0 = tail;
        Some(*ch)
    }

    pub fn consume_until(&mut self, ch: u8) -> Option<&[u8]> {
        let index = self.0.iter().position(|&c| c == ch)?;
        let (head, tail) = self.0.split_at(index);
        self.0 = &tail[1..];
        Some(head)
    }

    pub fn consume_expect(&mut self, ch: u8) -> Option<()> {
        if self.consume_if_equals(ch) {
            Some(())
        } else {
            None
        }
    }

    pub fn consume_if(&mut self, func: impl Fn(u8) -> bool) -> Option<u8> {
        if let Some(ch) = self.peek() {
            if func(ch) {
                self.0 = &self.0[1..];
                return Some(ch);
            }
        }
        None
    }

    pub fn consume_if_equals(&mut self, ch: u8) -> bool {
        self.consume_if(|x| x == ch).is_some()
    }

    pub fn consume_if_either(&mut self, options: &[u8]) -> Option<u8> {
        self.consume_if(|x| options.contains(&x))
    }

    pub fn skip_whitespace(&mut self) {
        while let Some((ch, tail)) = self.0.split_first() {
            if !ch.is_ascii_whitespace() {
                break;
            }
            self.0 = tail;
        }
    }
}

/// ```md
/// modifier        byte order      size        alignment
/// -----------------------------------------------------
/// @ (default)     native          native      native
/// ^               native          native      none
/// =               native          standard    none
/// <               little-endian   standard    none
/// > or !          big-endian      standard    none
/// ```
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Modifier {
    Native { is_aligned: bool },
    Standard { endian: Option<Endian> },
}

impl Default for Modifier {
    fn default() -> Self {
        Self::Native { is_aligned: true } // '@' is the implicit default
    }
}

impl Modifier {
    pub fn is_native_size(self) -> bool {
        matches!(self, Self::Native { .. })
    }

    pub fn is_standard_size(self) -> bool {
        matches!(self, Self::Standard { .. })
    }

    pub fn endian(self) -> Endian {
        match self {
            Self::Native { .. } => Endian::NATIVE,
            Self::Standard { endian, .. } => endian.unwrap_or(Endian::NATIVE),
        }
    }

    pub fn is_aligned(self) -> bool {
        match self {
            Self::Native { is_aligned: aligned } => aligned,
            Self::Standard { .. } => false,
        }
    }

    pub fn from_char(ch: u8) -> Option<Self> {
        Some(match ch {
            b'@' => Self::Native { is_aligned: true },
            b'^' => Self::Native { is_aligned: false },
            b'=' => Self::Standard { endian: None },
            b'<' => Self::Standard { endian: Some(Endian::Little) },
            b'>' | b'!' => Self::Standard { endian: Some(Endian::Big) },
            _ => return None,
        })
    }
}

fn integer_type<T: Default + Not<Output = T> + PartialOrd>(
    modifier: Modifier, standard_size: IntegerSize,
) -> Scalar {
    let size = if modifier.is_standard_size() {
        standard_size
    } else {
        match mem::size_of::<T>() {
            2 => IntegerSize::Two,
            4 => IntegerSize::Four,
            8 => IntegerSize::Eight,
            _ => unreachable!(),
        }
    };
    let signedness = Signedness::of::<T>();
    let endian = modifier.endian();
    Scalar::Integer(size, signedness, endian)
}

fn parse_scalar_type(s: &mut ByteStream, modifier: Modifier, itemsize: usize) -> Option<Scalar> {
    type S = Scalar;
    let endian = modifier.endian();

    Some(match s.consume_first()? {
        b's' => S::Bytes(itemsize),
        b'w' => S::Ucs4(itemsize),
        b'?' => S::Bool,
        b'b' => S::Char(Signedness::Signed),
        b'B' => S::Char(Signedness::Unsigned),
        b'h' => integer_type::<c_short>(modifier, IntegerSize::Two),
        b'H' => integer_type::<c_ushort>(modifier, IntegerSize::Two),
        b'i' => integer_type::<c_int>(modifier, IntegerSize::Four),
        b'I' => integer_type::<c_uint>(modifier, IntegerSize::Four),
        b'l' => integer_type::<c_long>(modifier, IntegerSize::Four),
        b'L' => integer_type::<c_ulong>(modifier, IntegerSize::Four),
        b'q' => integer_type::<c_longlong>(modifier, IntegerSize::Eight),
        b'Q' => integer_type::<c_ulonglong>(modifier, IntegerSize::Eight),
        b'e' => S::Float(FloatSize::Two, endian),
        b'f' => S::Float(FloatSize::Four, endian),
        b'd' => S::Float(FloatSize::Eight, endian),
        b'Z' => match s.consume_first()? {
            b'f' => S::Complex(ComplexSize::Four, endian),
            b'd' => S::Complex(ComplexSize::Eight, endian),
            _ => return None,
        },
        b'c' => S::Bytes(1),
        _ => return None,
    })
}

#[inline]
fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

#[inline]
fn lcm(a: usize, b: usize) -> usize {
    a / gcd(a, b) * b
}

#[inline]
fn calc_padding(offset: usize, align: usize) -> usize {
    // an equivalent of Python's `(-offset) % alignment`
    // note: always 0 if align == 1
    (align - offset % align) % align
}

fn add_trailing_padding<T: ScalarDescriptor>(
    mut desc: TypeDescriptor<T>, padding: usize,
) -> TypeDescriptor<T> {
    if let TypeDescriptor::Record(ref mut rec) = desc {
        rec.itemsize += padding;
    } else {
        let desc_itemsize = desc.itemsize();
        let field = FieldDescriptor::new(desc, Some(b"f0".as_ref()), 0);
        desc = TypeDescriptor::record(vec![field], desc_itemsize + padding);
    }
    desc
}

/// Parses type descriptor from buffer format string.
///
/// Assumes [PEP 3118](https://www.python.org/dev/peps/pep-3118) convention
/// for buffer format strings. Largerly a replication of
/// `numpy.core.internal._dtype_from_pep3118()`, with a few minor adjustments
/// and bug fixes (most notably, regarding trailing padding for struct
/// types, but also being more restrictive in general and disallowing broken
/// format strings like those with unmatched braces).
pub fn parse_type_descriptor(s: impl AsRef<[u8]>) -> Option<TypeDescriptor<Scalar>> {
    let mut stream = ByteStream::new(s.as_ref());
    let (desc, _align) = parse_type_descriptor_impl(&mut stream, false)?;
    if stream.is_empty() {
        Some(desc)
    } else {
        None
    }
}

fn parse_type_descriptor_impl(
    stream: &mut ByteStream, is_subdtype: bool,
) -> Option<(TypeDescriptor<Scalar>, usize)> {
    /*
    Differences from numpy.core.internal._dtype_from_pep3118:
    - Always match T{} braces (numpy allows 'T{f' and 'f}' as valid format strings)
    - (TODO: disallow zero-size subarray shapes, zero repetitions and zero-length strings)
     */

    let mut modifier = Modifier::default();

    let mut common_alignment = 1;
    let mut offset = 0;
    let mut out_fields = Vec::<FieldDescriptor<Scalar>>::new();
    let mut out_itemsize = 0;

    while !stream.is_empty() {
        // skip all whitespace that's not inside field names
        stream.skip_whitespace();

        // if '}' is encountered, return immediately if we're parsing a subdtype
        if stream.consume_if_equals(b'}') {
            if !is_subdtype {
                return None;
            } else {
                break;
            }
        }

        // parse optional subarray shape
        //
        // TODO: are multiple subarray shape prefixes allowed?
        // (see https://github.com/numpy/numpy/issues/20802)
        let shape = if stream.consume_if_equals(b'(') {
            let mut shape = vec![stream.parse_usize()?]; // TODO: zero size check
            while !stream.is_empty() {
                if stream.consume_if_equals(b')') {
                    break;
                }
                stream.consume_expect(b',')?;
                shape.push(stream.parse_usize()?); // TODO: zero size check
            }
            Some(shape)
        } else {
            None
        };

        // parse optional byte order modifier
        if let Some(ch) = stream.peek() {
            if let Some(m) = Modifier::from_char(ch) {
                modifier = m;
                stream.advance(1);
            }
        }

        // parse optional item sizes (defaults to 1)
        let mut itemsize = 1;
        if let Some(ch) = stream.peek() {
            if ch.is_ascii_digit() {
                itemsize = stream.parse_usize()?;
            }
        }

        // parse the actual type (or padding)
        let ((mut desc, align), is_padding) = if stream.consume_if_equals(b'x') {
            // padding
            ((TypeDescriptor::scalar(Scalar::Bool), 1), true)
        } else if stream.consume_if_equals(b'T') {
            // structured type
            if !stream.consume_if_equals(b'{') {
                return None;
            }
            (parse_type_descriptor_impl(stream, true)?, false)
        } else {
            // simple type
            let ty = parse_scalar_type(stream, modifier, itemsize)?;
            if ty.is_array_like() {
                itemsize = 1;
            }
            ((TypeDescriptor::scalar(ty), ty.alignment()), false)
        };

        // Native alignment may require padding (same logic as in numpy)
        // (Here we assume that the presence of a '@' character implicitly implies
        // that the start of the array is *already* aligned)
        let mut extra_offset = 0;
        if modifier.is_aligned() && align > 1 {
            common_alignment = lcm(align, common_alignment);

            let start_padding = calc_padding(offset, align);
            let intra_padding = calc_padding(desc.itemsize(), align);
            offset += start_padding;

            if intra_padding != 0 {
                let subarray_size = shape.as_ref().map(|shape| shape.iter().product::<usize>());
                if itemsize > 1 || subarray_size.map_or(false, |size| size > 1) {
                    // Inject internal padding to the end of the sub-item
                    desc = add_trailing_padding(desc, intra_padding);
                } else {
                    // We can postpone the injection of internal padding,
                    // as the item appears at most once
                    extra_offset = intra_padding;
                }
            }
        }

        // parse the optional field name
        let name = if stream.consume_if_equals(b':') {
            Some(stream.consume_until(b':')?) // TODO: empty field names?
        } else {
            None
        };

        // `struct` docs explicitly say that repeat-0 elements are for padding or
        // alignment - we further interpret this applies only to unnamed fields
        // (see https://github.com/numpy/numpy/pull/7798)
        if name.is_none() && itemsize == 0 {
            offset += extra_offset;
            out_itemsize = offset;
            continue;
        }

        // convert itemsize to sub-array
        if itemsize != 1 {
            desc = TypeDescriptor::array(Box::new(desc), vec![itemsize]);
        }

        // handle sub-arrays
        if let Some(shape) = shape {
            desc = TypeDescriptor::array(Box::new(desc), shape);
        }

        // add the field to the field list (unless it's an unnamed padding field)
        let desc_itemsize = desc.itemsize();
        if !is_padding || name.is_some() {
            if let Some(name) = name {
                // check for duplicate field names
                if out_fields
                    .iter()
                    .filter_map(|field| field.name.as_ref())
                    .any(|field_name| field_name.as_ref() == name)
                {
                    return None;
                }
            }
            // TODO: should names be validated in any way? (e.g. \0, \n should be probably banned)
            out_fields.push(FieldDescriptor::new(desc, name.map(|name| name.to_vec()), offset));
        }

        offset += desc_itemsize + extra_offset;
        out_itemsize = offset;
    }

    // extra final padding for aligned types
    //
    // See these for potential bugs and fixes in numpy itself:
    // - https://github.com/numpy/numpy/issues/7797
    // - https://github.com/numpy/numpy/pull/7798
    //
    // Inside of T{}, if in aligned mode, we add trailing padding like in a
    // C struct so the end of the struct is aligned.
    // Note that this behavior is *not* described by the PEP3118 spec, which
    // does not say anything about T{} trailing padding. Note also that the Py
    // struct docs say that trailing padding should *not* be implicitly added
    // when outside of T{}, and the user should explicitly add a 0-sized
    // trailing field to add padding, however struct does not implement T{}. So,
    // here numpy is taking the initiative to specify how trailing padding works
    // inside T{}, while we mimic struct outside of T{}.
    if modifier.is_aligned() && is_subdtype {
        // note: in the master numpy branch as of 1.20, `is_subdtype` is not checked
        out_itemsize += calc_padding(offset, common_alignment);
    }

    // check if this was a simple 1-item type, and unwrap it
    if let [field] = out_fields.as_slice() {
        if field.name.is_none() && field.offset == 0 && !is_subdtype {
            return Some((out_fields.pop().unwrap().desc, common_alignment));
        }
    }

    // at this point, numpy replaces all unnamed fields with f%d, but we can always do it later

    let desc = TypeDescriptor::record(out_fields, out_itemsize);
    Some((desc, common_alignment))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple() {
        let s = "T{b:a:xxxi:b:3b:c:xi:d:}";
        assert_eq!(
            format!("{:?}", parse_type_descriptor(s).unwrap()),
            "[16]{0:'a'=i8, 4:'b'=<i32, 8:'c'=(i8;[3]), 12:'d'=<i32}"
        );
    }

    #[test]
    fn test_byte_stream() {
        for s in ["", "x", ":(", "98765432109876543210"] {
            let mut b = ByteStream::new(s.as_bytes());
            assert_eq!(b.parse_usize(), None);
            assert_eq!(&*b, s.as_bytes());
        }

        for (s, v, t) in
            [("0", 0, ""), ("12", 12, ""), ("1234567890abc", 1234567890, "abc"), ("000x", 0, "x")]
        {
            let mut b = ByteStream::new(s.as_bytes());
            assert_eq!(b.parse_usize(), Some(v));
            assert_eq!(&*b, t.as_bytes());
        }
    }

    #[test]
    fn test_parse_modifier() {
        for (ch, m) in [
            (b'@', Modifier::Native { is_aligned: true }),
            (b'^', Modifier::Native { is_aligned: false }),
            (b'=', Modifier::Standard { endian: None }),
            (b'<', Modifier::Standard { endian: Some(Endian::Little) }),
            (b'>', Modifier::Standard { endian: Some(Endian::Big) }),
            (b'!', Modifier::Standard { endian: Some(Endian::Big) }),
        ] {
            assert_eq!(Modifier::from_char(ch), Some(m));
        }
        for ch in 0..0xff {
            assert_eq!(Modifier::from_char(ch).is_some(), b"@^=<>!".contains(&ch));
        }
    }
}
