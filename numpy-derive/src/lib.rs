use proc_macro2::{Span, TokenStream};
use proc_macro_error::{abort, proc_macro_error};

use quote::{quote, quote_spanned};
use syn::spanned::Spanned;
use syn::{parse_macro_input, parse_quote, Data, DeriveInput, Fields, GenericParam};

#[proc_macro_derive(Record)]
#[proc_macro_error]
pub fn derive(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let mut generics = input.generics;
    for param in &mut generics.params {
        if let GenericParam::Type(ref mut type_param) = *param {
            type_param.bounds.push(parse_quote!(pyo3_type_desc::Element<numpy_type_desc::Scalar>));
        }
    }
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let name = input.ident;
    let type_desc = impl_type_desc(&input.data, name.span());
    // TODO: repr(transparent) (in input.attrs)

    quote!(
        unsafe impl #impl_generics
        pyo3_type_desc::Element<numpy_type_desc::Scalar>
        for #name #ty_generics #where_clause
        {
            #[allow(unused_imports)]
            fn type_descriptor() -> pyo3_type_desc::TypeDescriptor<numpy_type_desc::Scalar> {
                use pyo3_type_desc::{RecordDescriptor, FieldDescriptor, TypeDescriptor, Element};
                use numpy_type_desc::Scalar;
                use numpy_type_desc::memoffset::{offset_of, offset_of_tuple};
                #type_desc
            }
        }

    )
    .into()
}

fn impl_type_desc_struct(fields: &Fields) -> TokenStream {
    let fields = match fields {
        Fields::Named(ref fields) => {
            let q = fields.named.iter().map(|f| {
                let (name, ty) = (&f.ident, &f.ty);
                quote_spanned! { f.span() =>
                    FieldDescriptor::new(
                        <#ty as Element<Scalar>>::type_descriptor(),
                        Some(stringify!(#name)),
                        offset_of!(Self, #name),
                    )
                }
            });
            quote! { #(#q),* }
        }
        Fields::Unnamed(ref fields) => {
            let q = fields.unnamed.iter().enumerate().map(|(index, f)| {
                let ty = &f.ty;
                quote_spanned! { f.span() =>
                    FieldDescriptor::new(
                        <#ty as Element<Scalar>>::type_descriptor(),
                        None,
                        offset_of_tuple!(Self, #index),
                    )
                }
            });
            quote! { #(#q),* }
        }
        Fields::Unit => {
            // TODO: do we want to support unit structs?
            quote! {}
        }
    };
    quote! {
        let fields = vec![#fields];
        let itemsize = ::std::mem::size_of::<Self>();
        let alignment = ::std::mem::align_of::<Self>();
        TypeDescriptor::Record(RecordDescriptor::new(
            fields, itemsize, Some(alignment),
        ))
    }
}

fn impl_type_desc(data: &Data, span: Span) -> TokenStream {
    match *data {
        Data::Enum(_) => {
            // TODO: do we want to support simple numeric enums?
            unimplemented!("deriving numpy::Record for enums will be supported later")
        }
        Data::Union(_) => abort!(span, "numpy::Record cannot be derived for unions"),
        Data::Struct(ref data) => impl_type_desc_struct(&data.fields),
    }
}
