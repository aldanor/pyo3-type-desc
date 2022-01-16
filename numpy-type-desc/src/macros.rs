/// Macro for constructing arbitrary type descriptors.
#[macro_export(local_inner_macros)]
macro_rules! td {
    ($($tt:tt)+) => {
        td_impl!($($tt)+)
    };
}

#[macro_export(local_inner_macros)]
#[doc(hidden)]
macro_rules! td_impl {
    // scalar type helpers

    (@td) => {
        pyo3_type_desc::TypeDescriptor<crate::Scalar>
    };

    (@base $ident:ident) => {
        <td_impl!(@td)>::Scalar(crate::Scalar::Base(pyo3_type_desc::Scalar::$ident))
    };
    (@base $ident:ident $($expr:expr)*) => {
        <td_impl!(@td)>::Scalar(crate::Scalar::Base(pyo3_type_desc::Scalar::$ident($($expr),*)))
    };

    (@char $sign:ident) => {
        td_impl!(@base Char pyo3_type_desc::Signedness::$sign)
    };
    (@int $size:ident $sign:ident $endian:ident) => {
        td_impl!(
            @base Integer
            pyo3_type_desc::IntegerSize::$size
            pyo3_type_desc::Signedness::$sign
            pyo3_type_desc::Endian::$endian
        )
    };
    (@float $size:ident $endian:ident) => {
        td_impl!(
            @base Float
            pyo3_type_desc::FloatSize::$size
            pyo3_type_desc::Endian::$endian
        )
    };
    (@complex $size:ident $endian:ident) => {
        td_impl!(
            @base Complex
            pyo3_type_desc::ComplexSize::$size
            pyo3_type_desc::Endian::$endian
        )
    };
    (@bytes $size:expr) => {
        td_impl!(@base Bytes $size)
    };
    (@ucs4 $size:expr) => {
        td_impl!(@base Ucs4 $size)
    };

    (@datetime $ty:ident $unit:expr) => {
        <td_impl!(@td)>::Scalar(crate::Scalar::$ty($unit))
    };

    // object type

    (object) => { <td_impl!(@td)>::Object };
    (obj) => { td_impl!(object) };
    (O) => { td_impl!(object) };

    // scalar types

    (bool) => { td_impl!(@base Bool) };
    (?) => { td_impl!(bool) };

    (i8) => { td_impl!(@char Signed) };
    (u8) => { td_impl!(@char Unsigned) };

    (i16) => { td_impl!(=i16) };
    (=i16) => { td_impl!(@int Two Signed NATIVE) };
    (<i16) => { td_impl!(@int Two Signed Little) };
    (>i16) => { td_impl!(@int Two Signed Big) };

    (u16) => { td_impl!(=u16) };
    (=u16) => { td_impl!(@int Two Unsigned NATIVE) };
    (<u16) => { td_impl!(@int Two Unsigned Little) };
    (>u16) => { td_impl!(@int Two Unsigned Big) };

    (i32) => { td_impl!(=i32) };
    (=i32) => { td_impl!(@int Four Signed NATIVE) };
    (<i32) => { td_impl!(@int Four Signed Little) };
    (>i32) => { td_impl!(@int Four Signed Big) };

    (u32) => { td_impl!(=u32) };
    (=u32) => { td_impl!(@int Four Unsigned NATIVE) };
    (<u32) => { td_impl!(@int Four Unsigned Little) };
    (>u32) => { td_impl!(@int Four Unsigned Big) };

    (i64) => { td_impl!(=i64) };
    (=i64) => { td_impl!(@int Eight Signed NATIVE) };
    (<i64) => { td_impl!(@int Eight Signed Little) };
    (>i64) => { td_impl!(@int Eight Signed Big) };

    (u64) => { td_impl!(=u64) };
    (=u64) => { td_impl!(@int Eight Unsigned NATIVE) };
    (<u64) => { td_impl!(@int Eight Unsigned Little) };
    (>u64) => { td_impl!(@int Eight Unsigned Big) };

    (f32) => { td_impl!(=f32) };
    (=f32) => { td_impl!(@float Four NATIVE) };
    (<f32) => { td_impl!(@float Four Little) };
    (>f32) => { td_impl!(@float Four Big) };

    (f64) => { td_impl!(=f64) };
    (=f64) => { td_impl!(@float Eight NATIVE) };
    (<f64) => { td_impl!(@float Eight Little) };
    (>f64) => { td_impl!(@float Eight Big) };

    (c32) => { td_impl!(=c32) };
    (=c32) => { td_impl!(@complex Four NATIVE) };
    (<c32) => { td_impl!(@complex Four Little) };
    (>c32) => { td_impl!(@complex Four Big) };

    (c64) => { td_impl!(=c64) };
    (=c64) => { td_impl!(@complex Eight NATIVE) };
    (<c64) => { td_impl!(@complex Eight Little) };
    (>c64) => { td_impl!(@complex Eight Big) };

    (s[$size:expr]) => { td_impl!(@bytes $size) };

    (u[$size:expr]) => { td_impl!(@ucs4 $size) };

    // datetime types

    (@dtunit Y) => { crate::DatetimeUnit::Year };
    (@dtunit Y) => { crate::DatetimeUnit::Year };
    (@dtunit M) => { crate::DatetimeUnit::Month };
    (@dtunit W) => { crate::DatetimeUnit::Week };
    (@dtunit D) => { crate::DatetimeUnit::Day };
    (@dtunit h) => { crate::DatetimeUnit::Hour };
    (@dtunit m) => { crate::DatetimeUnit::Minute };
    (@dtunit s) => { crate::DatetimeUnit::Second };
    (@dtunit ms) => { crate::DatetimeUnit::Millisecond };
    (@dtunit us) => { crate::DatetimeUnit::Microsecond };
    (@dtunit ns) => { crate::DatetimeUnit::Nanosecond };
    (@dtunit ps) => { crate::DatetimeUnit::Picosecond };
    (@dtunit fs) => { crate::DatetimeUnit::Femtosecond };
    (@dtunit as) => { crate::DatetimeUnit::Attosecond };

    (datetime64[$unit:ident]) => { td_impl!(@datetime Datetime td_impl!(@dtunit $unit)) };
    (timedelta64[$unit:ident]) => { td_impl!(@datetime Timedelta td_impl!(@dtunit $unit)) };

    (M8[$unit:ident]) => { td_impl!(datetime64[$unit]) };
    (m8[$unit:ident]) => { td_impl!(timedelta64[$unit]) };

    // record - empty case

    (@rec [] ([$s:expr])) => {
        td_impl!(@rec [] [$s, None::<usize>])
    };
    (@rec [] ([$s:expr, $a:expr])) => {
        td_impl!(@rec [] [$s, Some($a)])
    };

    // record - parse the optional field name and the offset followed by '=>'

    (@rec [$($f:tt)*] ($k:tt : $o:expr => $($tt:tt)*)) => {
        td_impl!(@rec [$($f)*] [Some($k), $o] ($($tt)*))
    };
    (@rec [$($f:tt)*] ($o:expr => $($tt:tt)*)) => {
        td_impl!(@rec [$($f)*] [None::<&str>, $o] ($($tt)*))
    };

    // record - parse the field type

    (@rec [$($f:tt)*] [$k:expr, $o:expr] (=$ty:ident $($tt:tt)*)) => {
        td_impl!(@rec [$($f)*] [$k, $o, td_impl!(=$ty)] ($($tt)*))
    };
    (@rec [$($f:tt)*] [$k:expr, $o:expr] (<$ty:ident $($tt:tt)*)) => {
        td_impl!(@rec [$($f)*] [$k, $o, td_impl!(<$ty)] ($($tt)*))
    };
    (@rec [$($f:tt)*] [$k:expr, $o:expr] (>$ty:ident $($tt:tt)*)) => {
        td_impl!(@rec [$($f)*] [$k, $o, td_impl!(>$ty)] ($($tt)*))
    };
    (@rec [$($f:tt)*] [$k:expr, $o:expr] (? $($tt:tt)*)) => {
        td_impl!(@rec [$($f)*] [$k, $o, td_impl!(?)] ($($tt)*))
    };
    (@rec [$($f:tt)*] [$k:expr, $o:expr] (s[$n:expr] $($tt:tt)*)) => {
        td_impl!(@rec [$($f)*] [$k, $o, td_impl!(s[$n])] ($($tt)*))
    };
    (@rec [$($f:tt)*] [$k:expr, $o:expr] (u[$n:expr] $($tt:tt)*)) => {
        td_impl!(@rec [$($f)*] [$k, $o, td_impl!(u[$n])] ($($tt)*))
    };
    (@rec [$($f:tt)*] [$k:expr, $o:expr] (datetime64[$u:ident] $($tt:tt)*)) => {
        td_impl!(@rec [$($f)*] [$k, $o, td_impl!(datetime64[$u])] ($($tt)*))
    };
    (@rec [$($f:tt)*] [$k:expr, $o:expr] (timedelta64[$u:ident] $($tt:tt)*)) => {
        td_impl!(@rec [$($f)*] [$k, $o, td_impl!(timedelta64[$u])] ($($tt)*))
    };
    (@rec [$($f:tt)*] [$k:expr, $o:expr] (M8[$u:ident] $($tt:tt)*)) => {
        td_impl!(@rec [$($f)*] [$k, $o, td_impl!(M8[$u])] ($($tt)*))
    };
    (@rec [$($f:tt)*] [$k:expr, $o:expr] (m8[$u:ident] $($tt:tt)*)) => {
        td_impl!(@rec [$($f)*] [$k, $o, td_impl!(m8[$u])] ($($tt)*))
    };
    (@rec [$($f:tt)*] [$k:expr, $o:expr] ($ty:ident $($tt:tt)*)) => {
        td_impl!(@rec [$($f)*] [$k, $o, td_impl!($ty)] ($($tt)*))
    };
    (@rec [$($f:tt)*] [$k:expr, $o:expr] ([$($arr:tt)*] $($tt:tt)*)) => {
        td_impl!(@rec [$($f)*] [$k, $o, td_impl!([$($arr)*])] ($($tt)*))
    };
    (@rec [$($f:tt)*] [$k:expr, $o:expr] ({$($rec:tt)*} $($tt:tt)*)) => {
        td_impl!(@rec [$($f)*] [$k, $o, td_impl!({$($rec)*})] ($($tt)*))
    };

    // record - save the field and repeat or finish

    (@rec [$($f:tt)*] [$k:expr, $o:expr, $d:expr] (, [$s:expr])) => {
        td_impl!(@rec [$($f)*] [$k, $o, $d] ([$s]))
    };
    (@rec [$($f:tt)*] [$k:expr, $o:expr, $d:expr] (, [$s:expr, $a:expr])) => {
        td_impl!(@rec [$($f)*] [$k, $o, $d] ([$s, $a]))
    };
    (@rec [$($f:tt)*] [$k:expr, $o:expr, $d:expr] ([$s:expr])) => {
        td_impl!(@rec [$($f)* ($k, $o, $d)] [$s, None::<usize>])
    };
    (@rec [$($f:tt)*] [$k:expr, $o:expr, $d:expr] ([$s:expr, $a:expr])) => {
        td_impl!(@rec [$($f)* ($k, $o, $d)] [$s, Some($a)])
    };
    (@rec [$($f:tt)*] [$k:expr, $o:expr, $d:expr] (, $($tt:tt)*)) => {
        td_impl!(@rec [$($f)* ($k, $o, $d)] ($($tt)*))
    };

    // record - build the descriptor

    (@rec [$(($k:expr, $o:expr, $d:expr))*] [$s:expr, $a:expr]) => {{
        let fields: Vec<pyo3_type_desc::FieldDescriptor<_>> = vec![$(
            pyo3_type_desc::FieldDescriptor { desc: $d, name: ($k).map(|n| n.into()), offset: $o }
        ),*];
        <td_impl!(@td)>::Record(pyo3_type_desc::RecordDescriptor {
            fields: fields.into(), itemsize: $s, alignment: $a,
        })
    }};

    // subarray - [(x, y); desc] syntax, build the descriptor

    ([($($d:expr),*); $($tt:tt)+]) => {{
        let shape: Vec<usize> = vec![$($d),*];
        let desc = Box::new(td_impl!($($tt)+));
        <td_impl!(@td)>::Array(pyo3_type_desc::ArrayDescriptor {
            desc: desc.into(),
            shape: shape.into(),
        })
    }};

    // subarray - alternative [x, y; desc] syntax

    ([$($d:expr),+; $($tt:tt)+]) => {
        td_impl!([($($d),+); $($tt)+])
    };

    // record array - {...} syntax

    ({$($tt:tt)*}) => {
        td_impl!(@rec [] ($($tt)*))
    };
}
