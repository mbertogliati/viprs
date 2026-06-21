#![allow(clippy::unnecessary_wraps)]
// REASON: dispatcher constructors stay fallible-compatible with feature-gated registries.

use std::collections::{HashMap, VecDeque};

use crate::domain::{
    colorspace::{
        Cicp, Cmyk, ColorspaceId, Greyscale, Hsv, Lab, Lch, Oklab, Oklch, Rgb16, SRgb, ScRgb, Ucs,
        Xyz, Yxy,
    },
    error::BuildError,
    format::{BandFormatId, F32, U8, U16},
    op::DynOperation,
    ops::colour::{
        BwToSRgb, CicpColourPrimaries, CicpMatrixCoefficients, CicpProfile, CicpToScRgb,
        CmykToRgbOp, CmykToXyz, ColourConvertBridge, HsvToSRgb, LabToLch, LabToSRgb, LabToXyz,
        LchToLab, LchToUcs, OklabToOklch, OklabToXyz, OklchToOklab, Rgb16ToSRgb, RgbToCmykOp,
        SRgbToHsv, SRgbToLab, SRgbToRgb16, SRgbToScRgb, SRgbToXyz, ScRgbToBw, ScRgbToSRgb,
        ScRgbToXyz, UcsToLch, XyzToCmyk, XyzToLab, XyzToOklab, XyzToSRgb, XyzToScRgb, XyzToYxy,
        YxyToXyz, cicp2scrgb::CicpTransferCharacteristics,
    },
};

// NOTE: `ColourspaceDispatcher` is the runtime registry boundary for colour conversion graphs.
// The caller only knows `(from, to, BandFormatId)` after parsing image metadata, so the
// dispatcher must choose among heterogeneous `ColourConvertBridge<...>` implementations at
// runtime and return a uniform handle. The dynamic dispatch stops at graph assembly: each boxed
// op still executes monomorphized pixel kernels internally, so the vtable is not on the per-pixel
// hot path that GUIDELINES.md reserves for static dispatch.
type DynOpFactory = fn(u32, BandFormatId) -> Result<Box<dyn DynOperation>, BuildError>;

/// One directed runtime conversion edge between two colorspaces.
///
/// The dispatcher stores edges instead of concrete ops so it can search routes before building the
/// final dynamic conversion chain.
///
/// # Examples
/// ```rust
/// # use viprs_runtime::colour_dispatcher::ColourspaceDispatcher;
/// # use viprs_runtime::domain::colorspace::ColorspaceId;
/// let path = ColourspaceDispatcher::new().find_path(ColorspaceId::SRgb, ColorspaceId::Lab);
/// assert!(path.is_some());
/// ```
#[derive(Clone, Copy)]
pub struct ColourspaceEdge {
    /// Stores the `from` value for this item.
    pub from: ColorspaceId,
    /// Stores the `to` value for this item.
    pub to: ColorspaceId,
    factory: DynOpFactory,
}

impl ColourspaceEdge {
    /// Build the dynamic colour-conversion op represented by this edge.
    ///
    /// This delays monomorphized converter selection until the runtime route is known.
    ///
    /// # Examples
    /// ```rust
    /// # use viprs_runtime::{colour_dispatcher::ColourspaceDispatcher, domain::{colorspace::ColorspaceId, format::BandFormatId}};
    /// let edge = ColourspaceDispatcher::new()
    ///     .find_path(ColorspaceId::SRgb, ColorspaceId::Lab)
    ///     .unwrap()
    ///     .into_iter()
    ///     .next()
    ///     .unwrap();
    /// let op = edge.build(3, BandFormatId::U8).unwrap();
    /// assert_eq!(op.bands(), 3);
    /// ```
    pub fn build(
        self,
        bands: u32,
        current_format: BandFormatId,
    ) -> Result<Box<dyn DynOperation>, BuildError> {
        (self.factory)(bands, current_format)
    }
}

/// Runtime router for colour conversions between known colorspaces.
///
/// This type chooses a shortest conversion path and then builds the corresponding dynamic op chain
/// without putting `dyn` dispatch inside the per-pixel kernels.
///
/// # Examples
/// ```rust
/// # use viprs_runtime::{colour_dispatcher::ColourspaceDispatcher, domain::colorspace::ColorspaceId};
/// let dispatcher = ColourspaceDispatcher::new();
/// assert!(dispatcher.find_path(ColorspaceId::SRgb, ColorspaceId::Lab).is_some());
/// ```
pub struct ColourspaceDispatcher;

macro_rules! simple_factory {
    ($name:ident, $converter_expr:expr, $converter_ty:ty, $from:ty, $to:ty) => {
        fn $name(bands: u32, _: BandFormatId) -> Result<Box<dyn DynOperation>, BuildError> {
            Ok(Box::new(
                ColourConvertBridge::<$converter_ty, $from, $to>::new($converter_expr, bands),
            ))
        }
    };
}

simple_factory!(build_lab_to_srgb, LabToSRgb, LabToSRgb, Lab, SRgb);
simple_factory!(build_bw_to_srgb, BwToSRgb, BwToSRgb, Greyscale, SRgb);
simple_factory!(build_rgb16_to_srgb, Rgb16ToSRgb, Rgb16ToSRgb, Rgb16, SRgb);
simple_factory!(build_srgb_to_hsv, SRgbToHsv, SRgbToHsv, SRgb, Hsv);
simple_factory!(build_hsv_to_srgb, HsvToSRgb, HsvToSRgb, Hsv, SRgb);
simple_factory!(build_srgb_to_xyz, SRgbToXyz, SRgbToXyz, SRgb, Xyz);
simple_factory!(build_xyz_to_srgb, XyzToSRgb, XyzToSRgb, Xyz, SRgb);
simple_factory!(build_srgb_to_rgb16, SRgbToRgb16, SRgbToRgb16, SRgb, Rgb16);
simple_factory!(build_srgb_to_scrgb, SRgbToScRgb, SRgbToScRgb, SRgb, ScRgb);
simple_factory!(build_scrgb_to_srgb, ScRgbToSRgb, ScRgbToSRgb, ScRgb, SRgb);
simple_factory!(build_scrgb_to_bw, ScRgbToBw, ScRgbToBw, ScRgb, Greyscale);
simple_factory!(build_scrgb_to_xyz, ScRgbToXyz, ScRgbToXyz, ScRgb, Xyz);
simple_factory!(build_xyz_to_scrgb, XyzToScRgb, XyzToScRgb, Xyz, ScRgb);
simple_factory!(build_xyz_to_lab, XyzToLab, XyzToLab, Xyz, Lab);
simple_factory!(build_lab_to_xyz, LabToXyz, LabToXyz, Lab, Xyz);
simple_factory!(build_lab_to_lch, LabToLch, LabToLch, Lab, Lch);
simple_factory!(build_lch_to_lab, LchToLab, LchToLab, Lch, Lab);
simple_factory!(build_lch_to_ucs, LchToUcs, LchToUcs, Lch, Ucs);
simple_factory!(build_ucs_to_lch, UcsToLch, UcsToLch, Ucs, Lch);
simple_factory!(build_xyz_to_oklab, XyzToOklab, XyzToOklab, Xyz, Oklab);
simple_factory!(build_oklab_to_xyz, OklabToXyz, OklabToXyz, Oklab, Xyz);
simple_factory!(
    build_oklab_to_oklch,
    OklabToOklch,
    OklabToOklch,
    Oklab,
    Oklch
);
simple_factory!(
    build_oklch_to_oklab,
    OklchToOklab,
    OklchToOklab,
    Oklch,
    Oklab
);
simple_factory!(build_xyz_to_yxy, XyzToYxy, XyzToYxy, Xyz, Yxy);
simple_factory!(build_yxy_to_xyz, YxyToXyz, YxyToXyz, Yxy, Xyz);

#[inline]
const fn require_rgb_or_rgba(
    from: ColorspaceId,
    to: ColorspaceId,
    bands: u32,
) -> Result<(), BuildError> {
    if matches!(bands, 3 | 4) {
        Ok(())
    } else {
        Err(BuildError::InvalidColourConversionInput {
            from,
            to,
            bands,
            expected: "3 or 4 bands",
        })
    }
}

fn build_srgb_to_lab(bands: u32, _: BandFormatId) -> Result<Box<dyn DynOperation>, BuildError> {
    require_rgb_or_rgba(ColorspaceId::SRgb, ColorspaceId::Lab, bands)?;
    Ok(Box::new(ColourConvertBridge::<SRgbToLab, SRgb, Lab>::new(
        SRgbToLab, bands,
    )))
}

const fn default_cicp_router_profile() -> CicpProfile {
    CicpProfile::new(
        CicpColourPrimaries::Bt709,
        CicpTransferCharacteristics::Linear,
        CicpMatrixCoefficients::RgbIdentity,
        true,
    )
}

fn build_cicp_to_scrgb(
    bands: u32,
    current_format: BandFormatId,
) -> Result<Box<dyn DynOperation>, BuildError> {
    let profile = default_cicp_router_profile();

    match current_format {
        BandFormatId::U8 => Ok(Box::new(
            ColourConvertBridge::<CicpToScRgb<U8>, Cicp, ScRgb>::new(
                CicpToScRgb::<U8>::new(profile),
                bands,
            ),
        )),
        BandFormatId::U16 => Ok(Box::new(
            ColourConvertBridge::<CicpToScRgb<U16>, Cicp, ScRgb>::new(
                CicpToScRgb::<U16>::new(profile),
                bands,
            ),
        )),
        format => Err(BuildError::UnsupportedFormat {
            op: "cicp_to_scrgb",
            format,
        }),
    }
}

fn build_srgb_to_cmyk(
    bands: u32,
    current_format: BandFormatId,
) -> Result<Box<dyn DynOperation>, BuildError> {
    match current_format {
        BandFormatId::U8 => Ok(Box::new(
            ColourConvertBridge::<RgbToCmykOp<U8>, SRgb, Cmyk>::new(
                RgbToCmykOp::<U8>::new(),
                bands,
            ),
        )),
        BandFormatId::F32 => Ok(Box::new(
            ColourConvertBridge::<RgbToCmykOp<F32>, SRgb, Cmyk>::new(
                RgbToCmykOp::<F32>::new(),
                bands,
            ),
        )),
        format => Err(BuildError::UnsupportedFormat {
            op: "srgb_to_cmyk",
            format,
        }),
    }
}

fn build_cmyk_to_srgb(
    bands: u32,
    current_format: BandFormatId,
) -> Result<Box<dyn DynOperation>, BuildError> {
    match current_format {
        BandFormatId::U8 => Ok(Box::new(
            ColourConvertBridge::<CmykToRgbOp<U8>, Cmyk, SRgb>::new(
                CmykToRgbOp::<U8>::new(),
                bands,
            ),
        )),
        BandFormatId::F32 => Ok(Box::new(
            ColourConvertBridge::<CmykToRgbOp<F32>, Cmyk, SRgb>::new(
                CmykToRgbOp::<F32>::new(),
                bands,
            ),
        )),
        format => Err(BuildError::UnsupportedFormat {
            op: "cmyk_to_srgb",
            format,
        }),
    }
}

fn build_cmyk_to_xyz(
    bands: u32,
    current_format: BandFormatId,
) -> Result<Box<dyn DynOperation>, BuildError> {
    match current_format {
        BandFormatId::U8 => Ok(Box::new(
            ColourConvertBridge::<CmykToXyz<U8>, Cmyk, Xyz>::new(CmykToXyz::<U8>::new(), bands),
        )),
        BandFormatId::F32 => Ok(Box::new(
            ColourConvertBridge::<CmykToXyz<F32>, Cmyk, Xyz>::new(CmykToXyz::<F32>::new(), bands),
        )),
        format => Err(BuildError::UnsupportedFormat {
            op: "cmyk_to_xyz",
            format,
        }),
    }
}

fn build_xyz_to_cmyk(bands: u32, _: BandFormatId) -> Result<Box<dyn DynOperation>, BuildError> {
    Ok(Box::new(
        ColourConvertBridge::<XyzToCmyk<U8>, Xyz, Cmyk>::new(XyzToCmyk::<U8>::new(), bands),
    ))
}

const COLOURSPACE_EDGES: [ColourspaceEdge; 31] = [
    ColourspaceEdge {
        from: ColorspaceId::Cicp,
        to: ColorspaceId::ScRgb,
        factory: build_cicp_to_scrgb,
    },
    ColourspaceEdge {
        from: ColorspaceId::Greyscale,
        to: ColorspaceId::SRgb,
        factory: build_bw_to_srgb,
    },
    ColourspaceEdge {
        from: ColorspaceId::Rgb16,
        to: ColorspaceId::SRgb,
        factory: build_rgb16_to_srgb,
    },
    ColourspaceEdge {
        from: ColorspaceId::SRgb,
        to: ColorspaceId::Lab,
        factory: build_srgb_to_lab,
    },
    ColourspaceEdge {
        from: ColorspaceId::Lab,
        to: ColorspaceId::SRgb,
        factory: build_lab_to_srgb,
    },
    ColourspaceEdge {
        from: ColorspaceId::SRgb,
        to: ColorspaceId::Hsv,
        factory: build_srgb_to_hsv,
    },
    ColourspaceEdge {
        from: ColorspaceId::Hsv,
        to: ColorspaceId::SRgb,
        factory: build_hsv_to_srgb,
    },
    ColourspaceEdge {
        from: ColorspaceId::SRgb,
        to: ColorspaceId::Xyz,
        factory: build_srgb_to_xyz,
    },
    ColourspaceEdge {
        from: ColorspaceId::Xyz,
        to: ColorspaceId::SRgb,
        factory: build_xyz_to_srgb,
    },
    ColourspaceEdge {
        from: ColorspaceId::SRgb,
        to: ColorspaceId::ScRgb,
        factory: build_srgb_to_scrgb,
    },
    ColourspaceEdge {
        from: ColorspaceId::SRgb,
        to: ColorspaceId::Rgb16,
        factory: build_srgb_to_rgb16,
    },
    ColourspaceEdge {
        from: ColorspaceId::ScRgb,
        to: ColorspaceId::SRgb,
        factory: build_scrgb_to_srgb,
    },
    ColourspaceEdge {
        from: ColorspaceId::ScRgb,
        to: ColorspaceId::Greyscale,
        factory: build_scrgb_to_bw,
    },
    ColourspaceEdge {
        from: ColorspaceId::ScRgb,
        to: ColorspaceId::Xyz,
        factory: build_scrgb_to_xyz,
    },
    ColourspaceEdge {
        from: ColorspaceId::Xyz,
        to: ColorspaceId::ScRgb,
        factory: build_xyz_to_scrgb,
    },
    ColourspaceEdge {
        from: ColorspaceId::Xyz,
        to: ColorspaceId::Lab,
        factory: build_xyz_to_lab,
    },
    ColourspaceEdge {
        from: ColorspaceId::Lab,
        to: ColorspaceId::Xyz,
        factory: build_lab_to_xyz,
    },
    ColourspaceEdge {
        from: ColorspaceId::Lab,
        to: ColorspaceId::Lch,
        factory: build_lab_to_lch,
    },
    ColourspaceEdge {
        from: ColorspaceId::Lch,
        to: ColorspaceId::Lab,
        factory: build_lch_to_lab,
    },
    ColourspaceEdge {
        from: ColorspaceId::Lch,
        to: ColorspaceId::Ucs,
        factory: build_lch_to_ucs,
    },
    ColourspaceEdge {
        from: ColorspaceId::Ucs,
        to: ColorspaceId::Lch,
        factory: build_ucs_to_lch,
    },
    ColourspaceEdge {
        from: ColorspaceId::SRgb,
        to: ColorspaceId::Cmyk,
        factory: build_srgb_to_cmyk,
    },
    ColourspaceEdge {
        from: ColorspaceId::Cmyk,
        to: ColorspaceId::SRgb,
        factory: build_cmyk_to_srgb,
    },
    ColourspaceEdge {
        from: ColorspaceId::Cmyk,
        to: ColorspaceId::Xyz,
        factory: build_cmyk_to_xyz,
    },
    ColourspaceEdge {
        from: ColorspaceId::Xyz,
        to: ColorspaceId::Cmyk,
        factory: build_xyz_to_cmyk,
    },
    ColourspaceEdge {
        from: ColorspaceId::Xyz,
        to: ColorspaceId::Oklab,
        factory: build_xyz_to_oklab,
    },
    ColourspaceEdge {
        from: ColorspaceId::Oklab,
        to: ColorspaceId::Xyz,
        factory: build_oklab_to_xyz,
    },
    ColourspaceEdge {
        from: ColorspaceId::Oklab,
        to: ColorspaceId::Oklch,
        factory: build_oklab_to_oklch,
    },
    ColourspaceEdge {
        from: ColorspaceId::Oklch,
        to: ColorspaceId::Oklab,
        factory: build_oklch_to_oklab,
    },
    ColourspaceEdge {
        from: ColorspaceId::Xyz,
        to: ColorspaceId::Yxy,
        factory: build_xyz_to_yxy,
    },
    ColourspaceEdge {
        from: ColorspaceId::Yxy,
        to: ColorspaceId::Xyz,
        factory: build_yxy_to_xyz,
    },
];

impl ColourspaceDispatcher {
    #[must_use]
    /// Create a dispatcher over the built-in conversion registry.
    ///
    /// The dispatcher is zero-sized because all routing data lives in static tables.
    ///
    /// # Examples
    /// ```rust
    /// # use viprs_runtime::colour_dispatcher::ColourspaceDispatcher;
    /// let dispatcher = ColourspaceDispatcher::new();
    /// let _ = dispatcher;
    /// ```
    pub const fn new() -> Self {
        Self
    }

    #[must_use]
    /// Find the shortest registered conversion path between two colorspaces.
    ///
    /// This lets callers validate that a route exists before materializing any dynamic ops.
    ///
    /// # Examples
    /// ```rust
    /// # use viprs_runtime::{colour_dispatcher::ColourspaceDispatcher, domain::colorspace::ColorspaceId};
    /// let path = ColourspaceDispatcher::new().find_path(ColorspaceId::SRgb, ColorspaceId::Lab);
    /// assert!(path.is_some());
    /// ```
    pub fn find_path(self, from: ColorspaceId, to: ColorspaceId) -> Option<Vec<ColourspaceEdge>> {
        if from == to {
            return Some(Vec::new());
        }

        let mut queue = VecDeque::from([from]);
        let mut previous = HashMap::<ColorspaceId, (ColorspaceId, usize)>::new();
        previous.insert(from, (from, usize::MAX));

        while let Some(current) = queue.pop_front() {
            for (edge_index, edge) in COLOURSPACE_EDGES.iter().copied().enumerate() {
                if edge.from != current || previous.contains_key(&edge.to) {
                    continue;
                }

                previous.insert(edge.to, (current, edge_index));
                if edge.to == to {
                    return reconstruct_path(from, to, &previous);
                }
                queue.push_back(edge.to);
            }
        }

        None
    }

    /// Build the dynamic colour-conversion ops for the shortest route between two colorspaces.
    ///
    /// This translates a routing decision into executable pipeline nodes while updating runtime
    /// band and format information step by step.
    ///
    /// # Examples
    /// ```rust
    /// # use viprs_runtime::{colour_dispatcher::ColourspaceDispatcher, domain::{colorspace::ColorspaceId, format::BandFormatId}};
    /// let ops = ColourspaceDispatcher::new()
    ///     .build_path(ColorspaceId::SRgb, ColorspaceId::Lab, 3, BandFormatId::U8)
    ///     .unwrap()
    ///     .unwrap();
    /// assert!(!ops.is_empty());
    /// ```
    pub fn build_path(
        self,
        from: ColorspaceId,
        to: ColorspaceId,
        bands: u32,
        current_format: BandFormatId,
    ) -> Result<Option<Vec<Box<dyn DynOperation>>>, BuildError> {
        // NOTE: The returned path is heterogeneous by construction, so the dispatcher erases each
        // concrete colour op exactly once while assembling the runtime-selected route.
        let Some(path) = self.find_path(from, to) else {
            return Ok(None);
        };

        let mut ops = Vec::with_capacity(path.len());
        let mut current_bands = bands;
        let mut current_format = current_format;

        for edge in path {
            let op = edge.build(current_bands, current_format)?;
            current_bands = op.bands();
            current_format = op.output_format();
            ops.push(op);
        }

        Ok(Some(ops))
    }
}

impl Default for ColourspaceDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

fn reconstruct_path(
    from: ColorspaceId,
    to: ColorspaceId,
    previous: &HashMap<ColorspaceId, (ColorspaceId, usize)>,
) -> Option<Vec<ColourspaceEdge>> {
    let mut path = Vec::new();
    let mut current = to;

    while current != from {
        let &(prior, edge_index) = previous.get(&current)?;
        path.push(COLOURSPACE_EDGES[edge_index]);
        current = prior;
    }

    path.reverse();
    Some(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::format::BandFormatId;

    #[test]
    fn identity_path_is_empty() {
        let path = ColourspaceDispatcher::new()
            .find_path(ColorspaceId::SRgb, ColorspaceId::SRgb)
            .expect("identity path must exist");

        assert!(path.is_empty());
    }

    #[test]
    fn srgb_to_lab_prefers_direct_conversion() {
        let path = ColourspaceDispatcher::new()
            .find_path(ColorspaceId::SRgb, ColorspaceId::Lab)
            .expect("sRGB → Lab path must exist");

        assert_eq!(path.len(), 1);
        assert_eq!(path[0].from, ColorspaceId::SRgb);
        assert_eq!(path[0].to, ColorspaceId::Lab);
    }

    #[test]
    fn srgb_to_hsv_finds_a_path() {
        let path = ColourspaceDispatcher::new()
            .find_path(ColorspaceId::SRgb, ColorspaceId::Hsv)
            .expect("sRGB → HSV path must exist");

        assert!(!path.is_empty());
        assert_eq!(path.first().map(|edge| edge.from), Some(ColorspaceId::SRgb));
        assert_eq!(path.last().map(|edge| edge.to), Some(ColorspaceId::Hsv));
    }

    #[test]
    fn scrgb_to_bw_prefers_direct_conversion() {
        let path = ColourspaceDispatcher::new()
            .find_path(ColorspaceId::ScRgb, ColorspaceId::Greyscale)
            .expect("scRGB → greyscale path must exist");

        assert_eq!(
            path.iter()
                .map(|edge| (edge.from, edge.to))
                .collect::<Vec<_>>(),
            vec![(ColorspaceId::ScRgb, ColorspaceId::Greyscale)]
        );
    }

    #[test]
    fn bw_to_srgb_prefers_direct_conversion() {
        let path = ColourspaceDispatcher::new()
            .find_path(ColorspaceId::Greyscale, ColorspaceId::SRgb)
            .expect("greyscale → sRGB path must exist");

        assert_eq!(
            path.iter()
                .map(|edge| (edge.from, edge.to))
                .collect::<Vec<_>>(),
            vec![(ColorspaceId::Greyscale, ColorspaceId::SRgb)]
        );
    }

    #[test]
    fn rgb16_to_srgb_prefers_direct_conversion() {
        let path = ColourspaceDispatcher::new()
            .find_path(ColorspaceId::Rgb16, ColorspaceId::SRgb)
            .expect("rgb16 → sRGB path must exist");

        assert_eq!(
            path.iter()
                .map(|edge| (edge.from, edge.to))
                .collect::<Vec<_>>(),
            vec![(ColorspaceId::Rgb16, ColorspaceId::SRgb)]
        );
    }

    #[test]
    fn xyz_to_bw_routes_via_scrgb() {
        let path = ColourspaceDispatcher::new()
            .find_path(ColorspaceId::Xyz, ColorspaceId::Greyscale)
            .expect("XYZ → greyscale path must exist");

        assert_eq!(
            path.iter()
                .map(|edge| (edge.from, edge.to))
                .collect::<Vec<_>>(),
            vec![
                (ColorspaceId::Xyz, ColorspaceId::ScRgb),
                (ColorspaceId::ScRgb, ColorspaceId::Greyscale),
            ]
        );
    }

    #[test]
    fn lab_to_bw_routes_via_scrgb() {
        let path = ColourspaceDispatcher::new()
            .find_path(ColorspaceId::Lab, ColorspaceId::Greyscale)
            .expect("Lab → greyscale path must exist");

        assert_eq!(
            path.last().map(|edge| (edge.from, edge.to)),
            Some((ColorspaceId::ScRgb, ColorspaceId::Greyscale))
        );
        assert_eq!(path.first().map(|edge| edge.from), Some(ColorspaceId::Lab));
        assert!(path.iter().any(|edge| edge.to == ColorspaceId::ScRgb));
    }

    #[test]
    fn cicp_to_scrgb_builds_router_edge() {
        let ops = ColourspaceDispatcher::new()
            .build_path(
                ColorspaceId::Cicp,
                ColorspaceId::ScRgb,
                3,
                BandFormatId::U16,
            )
            .expect("CICP → scRGB edge must build")
            .expect("CICP → scRGB path must exist");

        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].output_format(), BandFormatId::F32);
        assert_eq!(ops[0].output_colorspace(), Some(ColorspaceId::ScRgb));
    }

    #[test]
    fn srgb_to_lab_rejects_one_band_input() {
        let result = ColourspaceDispatcher::new().build_path(
            ColorspaceId::SRgb,
            ColorspaceId::Lab,
            1,
            BandFormatId::U8,
        );

        assert!(matches!(
            result,
            Err(BuildError::InvalidColourConversionInput {
                from: ColorspaceId::SRgb,
                to: ColorspaceId::Lab,
                bands: 1,
                expected: "3 or 4 bands",
            })
        ));
    }
}
