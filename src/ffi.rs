#![allow(clippy::missing_safety_doc)]

//! C-compatible entry points for moving raw pixel buffers into simple viprs image handles.

/// Owns a pixel buffer and its basic shape metadata for the C FFI surface.
///
/// This type solves cross-language image ownership by bundling raw bytes with dimensions inside a
/// single opaque handle that C callers can pass back to viprs accessors and destructors.
///
/// # Examples
/// ```rust
/// # #[cfg(feature = "ffi")]
/// # fn main() {
/// use viprs::ffi::{viprs_image_free, viprs_image_new};
///
/// let pixels = [255_u8, 0, 0];
/// let image = unsafe { viprs_image_new(pixels.as_ptr(), pixels.len(), 1, 1, 3) };
/// assert!(!image.is_null());
/// unsafe { viprs_image_free(image) };
/// # }
/// # #[cfg(not(feature = "ffi"))]
/// # fn main() {}
/// ```
pub struct ViprsImage {
    pub(crate) data: Vec<u8>,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) bands: u8,
}

fn expected_len(width: u32, height: u32, bands: u8) -> Option<usize> {
    let pixels = usize::try_from(width)
        .ok()?
        .checked_mul(usize::try_from(height).ok()?)?;
    pixels.checked_mul(usize::from(bands))
}

/// Creates an owned FFI image handle from raw bytes.
///
/// This function solves transfer of pixel buffers from C into viprs by validating the supplied
/// shape and length before copying the bytes into an owned allocation.
///
/// # Examples
/// ```rust
/// # #[cfg(feature = "ffi")]
/// # fn main() {
/// use viprs::ffi::{viprs_image_free, viprs_image_new};
///
/// let pixels = [255_u8, 0, 0];
/// let image = unsafe { viprs_image_new(pixels.as_ptr(), pixels.len(), 1, 1, 3) };
///
/// assert!(!image.is_null());
/// unsafe { viprs_image_free(image) };
/// # }
/// # #[cfg(not(feature = "ffi"))]
/// # fn main() {}
/// ```
#[unsafe(no_mangle)]
pub unsafe extern "C" fn viprs_image_new(
    data: *const u8,
    len: usize,
    width: u32,
    height: u32,
    bands: u8,
) -> *mut ViprsImage {
    if data.is_null() {
        return std::ptr::null_mut();
    }

    let Some(required_len) = expected_len(width, height, bands) else {
        return std::ptr::null_mut();
    };

    if len != required_len {
        return std::ptr::null_mut();
    }

    // SAFETY: The caller guarantees that `data` points to `len` initialized bytes.
    let slice = unsafe { std::slice::from_raw_parts(data, len) };
    let img = Box::new(ViprsImage {
        data: slice.to_vec(),
        width,
        height,
        bands,
    });
    Box::into_raw(img)
}

/// Releases an image handle created by [`viprs_image_new`].
///
/// This function solves FFI ownership cleanup by letting external callers return the heap
/// allocation that backs a previously created `ViprsImage`.
///
/// # Examples
/// ```rust
/// # #[cfg(feature = "ffi")]
/// # fn main() {
/// use viprs::ffi::{viprs_image_free, viprs_image_new};
///
/// let pixels = [1_u8];
/// let image = unsafe { viprs_image_new(pixels.as_ptr(), pixels.len(), 1, 1, 1) };
/// unsafe { viprs_image_free(image) };
/// # }
/// # #[cfg(not(feature = "ffi"))]
/// # fn main() {}
/// ```
#[unsafe(no_mangle)]
pub unsafe extern "C" fn viprs_image_free(img: *mut ViprsImage) {
    if img.is_null() {
        return;
    }

    // SAFETY: `img` was returned by `viprs_image_new` and has not been freed yet.
    unsafe { drop(Box::from_raw(img)) };
}

/// Returns the width stored in an FFI image handle.
///
/// This accessor solves metadata inspection from C without exposing the internal Rust struct
/// layout directly to foreign callers.
///
/// # Examples
/// ```rust
/// # #[cfg(feature = "ffi")]
/// # fn main() {
/// use viprs::ffi::{viprs_image_free, viprs_image_new, viprs_image_width};
///
/// let pixels = [1_u8, 2];
/// let image = unsafe { viprs_image_new(pixels.as_ptr(), pixels.len(), 2, 1, 1) };
/// assert_eq!(unsafe { viprs_image_width(image) }, 2);
/// unsafe { viprs_image_free(image) };
/// # }
/// # #[cfg(not(feature = "ffi"))]
/// # fn main() {}
/// ```
#[unsafe(no_mangle)]
pub unsafe extern "C" fn viprs_image_width(img: *const ViprsImage) -> u32 {
    if img.is_null() {
        return 0;
    }

    // SAFETY: `img` is checked for null above and is expected to point to a valid `ViprsImage`.
    unsafe { (*img).width }
}

/// Returns the height stored in an FFI image handle.
///
/// This accessor solves foreign-language metadata lookup for callers that need image dimensions
/// before iterating over or exporting the pixel buffer.
///
/// # Examples
/// ```rust
/// # #[cfg(feature = "ffi")]
/// # fn main() {
/// use viprs::ffi::{viprs_image_free, viprs_image_height, viprs_image_new};
///
/// let pixels = [1_u8, 2];
/// let image = unsafe { viprs_image_new(pixels.as_ptr(), pixels.len(), 1, 2, 1) };
/// assert_eq!(unsafe { viprs_image_height(image) }, 2);
/// unsafe { viprs_image_free(image) };
/// # }
/// # #[cfg(not(feature = "ffi"))]
/// # fn main() {}
/// ```
#[unsafe(no_mangle)]
pub unsafe extern "C" fn viprs_image_height(img: *const ViprsImage) -> u32 {
    if img.is_null() {
        return 0;
    }

    // SAFETY: `img` is checked for null above and is expected to point to a valid `ViprsImage`.
    unsafe { (*img).height }
}

/// Returns the number of bands stored in an FFI image handle.
///
/// This accessor solves channel-count inspection for foreign callers that need to interpret the
/// raw pixel layout correctly.
///
/// # Examples
/// ```rust
/// # #[cfg(feature = "ffi")]
/// # fn main() {
/// use viprs::ffi::{viprs_image_bands, viprs_image_free, viprs_image_new};
///
/// let pixels = [255_u8, 0, 0];
/// let image = unsafe { viprs_image_new(pixels.as_ptr(), pixels.len(), 1, 1, 3) };
/// assert_eq!(unsafe { viprs_image_bands(image) }, 3);
/// unsafe { viprs_image_free(image) };
/// # }
/// # #[cfg(not(feature = "ffi"))]
/// # fn main() {}
/// ```
#[unsafe(no_mangle)]
pub unsafe extern "C" fn viprs_image_bands(img: *const ViprsImage) -> u8 {
    if img.is_null() {
        return 0;
    }

    // SAFETY: `img` is checked for null above and is expected to point to a valid `ViprsImage`.
    unsafe { (*img).bands }
}

/// Returns a read-only pointer to the stored pixel buffer.
///
/// This accessor solves zero-copy reads from foreign languages by exposing the owned buffer
/// pointer while keeping mutation and lifetime management on the Rust side.
///
/// # Examples
/// ```rust
/// # #[cfg(feature = "ffi")]
/// # fn main() {
/// use viprs::ffi::{viprs_image_data, viprs_image_free, viprs_image_new};
///
/// let pixels = [9_u8, 8, 7];
/// let image = unsafe { viprs_image_new(pixels.as_ptr(), pixels.len(), 3, 1, 1) };
/// let ptr = unsafe { viprs_image_data(image) };
///
/// assert!(!ptr.is_null());
/// unsafe { viprs_image_free(image) };
/// # }
/// # #[cfg(not(feature = "ffi"))]
/// # fn main() {}
/// ```
#[unsafe(no_mangle)]
pub unsafe extern "C" fn viprs_image_data(img: *const ViprsImage) -> *const u8 {
    if img.is_null() {
        return std::ptr::null();
    }

    // SAFETY: `img` is checked for null above and is expected to point to a valid `ViprsImage`.
    unsafe { (*img).data.as_ptr() }
}

#[cfg(test)]
mod tests {
    use super::{
        viprs_image_bands, viprs_image_data, viprs_image_free, viprs_image_height, viprs_image_new,
        viprs_image_width,
    };

    #[test]
    fn test_ffi_roundtrip() {
        let data = vec![255_u8, 0, 0, 0, 255, 0];

        // SAFETY: The pointers passed to and returned from the FFI are derived from valid test data.
        unsafe {
            let img = viprs_image_new(data.as_ptr(), data.len(), 2, 1, 3);
            assert!(!img.is_null());
            assert_eq!(viprs_image_width(img), 2);
            assert_eq!(viprs_image_height(img), 1);
            assert_eq!(viprs_image_bands(img), 3);

            let ffi_data = viprs_image_data(img);
            assert!(!ffi_data.is_null());
            // SAFETY: `ffi_data` points into the image allocation for exactly `data.len()` bytes.
            let ffi_slice = std::slice::from_raw_parts(ffi_data, data.len());
            assert_eq!(ffi_slice, data.as_slice());

            viprs_image_free(img);
        }
    }

    #[test]
    fn test_ffi_null_data() {
        // SAFETY: Passing a null pointer is the behavior under test for constructor validation.
        unsafe {
            let img = viprs_image_new(std::ptr::null(), 0, 0, 0, 0);
            assert!(img.is_null());
        }
    }

    #[test]
    fn test_ffi_rejects_invalid_length() {
        let data = [1_u8, 2, 3];

        // SAFETY: The raw pointer comes from a valid local buffer; length mismatch is the behavior under test.
        unsafe {
            let img = viprs_image_new(data.as_ptr(), data.len(), 2, 1, 3);
            assert!(img.is_null());
        }
    }

    #[test]
    fn test_ffi_null_image_accessors() {
        // SAFETY: Passing null handles is the behavior under test for the accessor guards.
        unsafe {
            assert_eq!(viprs_image_width(std::ptr::null()), 0);
            assert_eq!(viprs_image_height(std::ptr::null()), 0);
            assert_eq!(viprs_image_bands(std::ptr::null()), 0);
            assert!(viprs_image_data(std::ptr::null()).is_null());
            viprs_image_free(std::ptr::null_mut());
        }
    }
}
