//! Interpolation kernels for resampling operations.
//!
//! `InterpolationKernel` is an enum rather than a trait. An enum allows:
//! - Exhaustive match in SIMD dispatch paths — the compiler can specialise a
//!   hot inner loop per variant without runtime indirection.
//! - A closed, finite set of kernels that matches the libvips resample surface
//!   plus a handful of compatibility aliases already used internally.
//! - Zero-cost `Copy` storage inside `ReduceH`/`ReduceV` — no heap allocation.
//!

/// The set of supported interpolation kernels for resampling operations.
///
/// Keeping kernels as a closed enum lets planners and SIMD dispatchers specialize without runtime
/// trait objects.
///
/// # Examples
/// ```rust
/// # use viprs::domain::kernel::InterpolationKernel;
/// assert_eq!(InterpolationKernel::Lanczos3.window_size(), 6);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationKernel {
    /// Nearest-neighbour. Support = 0.5. No interpolation, fastest path.
    Nearest,
    /// Bilinear (tent function). Support = 1.0. Smooth but blurry at high reduction.
    Bilinear,
    /// libvips bicubic interpolator (Keys / Catmull-Rom). Support = 2.0.
    Bicubic,
    /// Interpolatory quadratic spline used by libvips-compatible affine sampling.
    Quadratic,
    /// libvips nohalo interpolation.
    ///
    /// `Affine` routes this kernel through the full nonlinear 2-D nohalo + LBB
    /// interpolator from `libvips/resample/nohalo.cpp`. The scalar
    /// `interpolate(x)` fallback remains a separable windowed-sinc approximation
    /// for callers that still require 1-D tap weights.
    Nohalo,
    /// Placeholder for libvips locally bounded bicubic interpolation.
    Lbb,
    /// libvips vertex-split quadratic B-splines interpolation.
    Vsqbs,
    /// Compatibility alias for the existing Catmull-Rom cubic path.
    CatmullRom,
    /// Lanczos with 2 lobes. Support = 2.0. Slightly softer than Lanczos3.
    Lanczos2,
    /// Lanczos with 3 lobes. Support = 3.0. Sharpest; ringing on hard edges.
    Lanczos3,
}

impl InterpolationKernel {
    #[inline]
    fn bicubic_weight(x: f64) -> f64 {
        let x = x.abs();
        if x < 1.0 {
            (2.5 * x).mul_add(-x, 1.5 * x * x * x) + 1.0
        } else if x < 2.0 {
            4.0f64.mul_add(-x, (2.5 * x).mul_add(x, -0.5 * x * x * x)) + 2.0
        } else {
            0.0
        }
    }

    /// Half-width of this kernel in output-pixel units.
    #[inline]
    #[must_use]
    pub const fn support(self) -> f64 {
        match self {
            Self::Nearest => 0.5,
            Self::Bilinear => 1.0,
            Self::Bicubic | Self::Lbb | Self::Vsqbs | Self::CatmullRom | Self::Lanczos2 => 2.0,
            Self::Quadratic => 1.5,
            Self::Nohalo => 2.5,
            Self::Lanczos3 => 3.0,
        }
    }

    /// Integer stencil width used by affine-family samplers.
    #[inline]
    #[must_use]
    pub const fn window_size(self) -> u32 {
        match self {
            Self::Nearest => 1,
            Self::Bilinear => 2,
            Self::Nohalo | Self::Lanczos3 => 6,
            Self::Bicubic
            | Self::Quadratic
            | Self::Lbb
            | Self::Vsqbs
            | Self::CatmullRom
            | Self::Lanczos2 => 4,
        }
    }

    /// Input samples read before the integer source coordinate in affine-family samplers.
    #[inline]
    #[must_use]
    pub const fn window_offset(self) -> i32 {
        match self {
            Self::Nearest | Self::Bilinear => 0,
            Self::Nohalo | Self::Lanczos3 => 2,
            Self::Bicubic
            | Self::Quadratic
            | Self::Lbb
            | Self::Vsqbs
            | Self::CatmullRom
            | Self::Lanczos2 => 1,
        }
    }

    /// Padding needed around mapped affine bounds.
    #[inline]
    #[must_use]
    pub const fn affine_padding(self) -> (i32, i32) {
        let left = self.window_offset();
        let right = self.window_size() as i32 - left - 1;
        (left, right)
    }

    /// Number of extra input pixels needed on each side of a tile when reducing by `factor`.
    #[inline]
    #[must_use]
    pub fn halo_for_factor(self, factor: f64) -> u32 {
        debug_assert!(
            factor >= 1.0,
            "halo_for_factor: factor must be >= 1.0 for reduce ops"
        );
        (self.support() / factor).ceil() as u32
    }

    /// Evaluate the kernel weight at normalised distance `x`.
    #[inline]
    #[must_use]
    pub fn interpolate(self, x: f64) -> f64 {
        match self {
            Self::Nearest => {
                if x.abs() < 0.5 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Bilinear => {
                let x = x.abs();
                if x < 1.0 { 1.0 - x } else { 0.0 }
            }
            Self::Bicubic | Self::CatmullRom => {
                let x = x.abs();
                if x < 1.0 {
                    (2.5 * x).mul_add(-x, 1.5 * x * x * x) + 1.0
                } else if x < 2.0 {
                    4.0f64.mul_add(-x, (2.5 * x).mul_add(x, -0.5 * x * x * x)) + 2.0
                } else {
                    0.0
                }
            }
            Self::Quadratic => {
                // libvips parity here uses the interpolatory quadratic spline with
                // support [-1.5, 1.5], not the quadratic B-spline smoother used by VSQBS.
                // See .libvips_repo/libvips/resample/quadratic.c for the corresponding
                // affine-family resample path that selects this kernel.
                let x = x.abs();
                if x < 0.5 {
                    (2.0 * x).mul_add(-x, 1.0)
                } else if x < 1.5 {
                    2.5f64.mul_add(-x, x * x) + 1.5
                } else {
                    0.0
                }
            }
            Self::Nohalo => {
                let x = x.abs();
                if x < f64::EPSILON {
                    1.0
                } else if x < 2.5 {
                    let pi_x = std::f64::consts::PI * x;
                    let sinc = pi_x.sin() / pi_x;
                    let window = 0.5 * (1.0 + (pi_x / 2.5).cos());
                    sinc * window
                } else {
                    0.0
                }
            }
            // True LBB is a nonlinear 2-D Hermite interpolator. `Affine` special-cases
            // it with the exact libvips stencil/limiter port; the 1-D fallback keeps
            // scalar kernel utilities well-defined for planning code.
            Self::Lbb => Self::bicubic_weight(x),
            // VSQBS remains available for affine-family utilities that need the
            // quadratic B-spline basis, but ReduceH/ReduceV reject it because
            // libvips never uses VSQBS on separable 1-D reduce paths.
            Self::Vsqbs => quadratic_b_spline_weight(x),
            Self::Lanczos2 => {
                let x = x.abs();
                if x < f64::EPSILON {
                    1.0
                } else if x < 2.0 {
                    let pi_x = std::f64::consts::PI * x;
                    2.0 * pi_x.sin() * (pi_x / 2.0).sin() / (pi_x * pi_x)
                } else {
                    0.0
                }
            }
            Self::Lanczos3 => {
                let x = x.abs();
                if x < f64::EPSILON {
                    1.0
                } else if x < 3.0 {
                    let pi_x = std::f64::consts::PI * x;
                    3.0 * pi_x.sin() * (pi_x / 3.0).sin() / (pi_x * pi_x)
                } else {
                    0.0
                }
            }
        }
    }

    /// Backwards-compatible alias for older call sites.
    #[inline]
    #[must_use]
    pub fn weight(self, x: f64) -> f64 {
        self.interpolate(x)
    }

    /// Evaluate a 4×4 affine interpolation neighborhood.
    ///
    /// `x_phase` and `y_phase` are the fractional parts of the sampling
    /// location in `[0, 1)`. `neighborhood` spans
    /// `[floor(x)-1..=floor(x)+2] × [floor(y)-1..=floor(y)+2]`.
    #[inline]
    #[allow(clippy::unreachable)]
    // REASON: callers route all other kernels through the 1-D separable interpolation path.
    #[must_use]
    pub fn interpolate_2d<const N: usize>(
        self,
        x_phase: f64,
        y_phase: f64,
        neighborhood: &[[f64; N]; N],
    ) -> f64 {
        match self {
            Self::Nohalo => interpolate_nohalo_2d(x_phase, y_phase, neighborhood),
            Self::Vsqbs => interpolate_vsqbs_2d(x_phase, y_phase, neighborhood),
            _ => unreachable!("interpolate_2d is only implemented for Nohalo and Vsqbs"),
        }
    }
}

#[inline]
fn quadratic_b_spline_weight(x: f64) -> f64 {
    let x = x.abs();
    if x < 0.5 {
        x.mul_add(-x, 0.75)
    } else if x < 1.5 {
        let edge = x - 1.5;
        0.5 * edge * edge
    } else {
        0.0
    }
}

#[inline]
fn nohalo_anchor_and_sign(phase: f64) -> (usize, i32, f64) {
    debug_assert!((0.0..1.0).contains(&phase), "phase must be in [0, 1)");

    if phase < 0.5 {
        (2, 1, phase)
    } else {
        (3, -1, phase - 1.0)
    }
}

#[inline]
const fn nohalo_index(anchor: usize, sign: i32, offset: i32) -> usize {
    (anchor as i32 + offset * sign) as usize
}

#[inline]
fn nohalo_min(x: f64, y: f64) -> f64 {
    if x <= y { x } else { y }
}

#[inline]
fn nohalo_max(x: f64, y: f64) -> f64 {
    if x >= y { x } else { y }
}

#[inline]
fn nohalo_minmod(a: f64, b: f64, a_times_a: f64, a_times_b: f64) -> f64 {
    if a_times_b >= 0.0 {
        if a_times_a <= a_times_b { a } else { b }
    } else {
        0.0
    }
}

#[inline]
fn interpolate_nohalo_2d<const N: usize>(
    x_phase: f64,
    y_phase: f64,
    neighborhood: &[[f64; N]; N],
) -> f64 {
    debug_assert!(N >= 6, "nohalo requires a 6x6 neighborhood");

    let (anchor_x, sign_x, x_0) = nohalo_anchor_and_sign(x_phase);
    let (anchor_y, sign_y, y_0) = nohalo_anchor_and_sign(y_phase);
    let stencil = nohalo_subdivision(neighborhood, anchor_x, sign_x, anchor_y, sign_y);

    let twice_abs_x_0 = f64::from(2 * sign_x) * x_0;
    let twice_abs_y_0 = f64::from(2 * sign_y) * y_0;

    nohalo_lbbicubic(&stencil, twice_abs_x_0, twice_abs_y_0)
}

#[inline]
fn nohalo_subdivision<const N: usize>(
    neighborhood: &[[f64; N]; N],
    anchor_x: usize,
    sign_x: i32,
    anchor_y: usize,
    sign_y: i32,
) -> [f64; 16] {
    let row_uno = nohalo_index(anchor_y, sign_y, -2);
    let row_dos = nohalo_index(anchor_y, sign_y, -1);
    let row_tre = anchor_y;
    let row_qua = nohalo_index(anchor_y, sign_y, 1);
    let row_cin = nohalo_index(anchor_y, sign_y, 2);
    let col_one = nohalo_index(anchor_x, sign_x, -2);
    let col_two = nohalo_index(anchor_x, sign_x, -1);
    let col_thr = anchor_x;
    let col_fou = nohalo_index(anchor_x, sign_x, 1);
    let col_fiv = nohalo_index(anchor_x, sign_x, 2);

    let uno_two = neighborhood[row_uno][col_two];
    let uno_thr = neighborhood[row_uno][col_thr];
    let uno_fou = neighborhood[row_uno][col_fou];
    let dos_one = neighborhood[row_dos][col_one];
    let dos_two = neighborhood[row_dos][col_two];
    let dos_thr = neighborhood[row_dos][col_thr];
    let dos_fou = neighborhood[row_dos][col_fou];
    let dos_fiv = neighborhood[row_dos][col_fiv];
    let tre_one = neighborhood[row_tre][col_one];
    let tre_two = neighborhood[row_tre][col_two];
    let tre_thr = neighborhood[row_tre][col_thr];
    let tre_fou = neighborhood[row_tre][col_fou];
    let tre_fiv = neighborhood[row_tre][col_fiv];
    let qua_one = neighborhood[row_qua][col_one];
    let qua_two = neighborhood[row_qua][col_two];
    let qua_thr = neighborhood[row_qua][col_thr];
    let qua_fou = neighborhood[row_qua][col_fou];
    let qua_fiv = neighborhood[row_qua][col_fiv];
    let cin_two = neighborhood[row_cin][col_two];
    let cin_thr = neighborhood[row_cin][col_thr];
    let cin_fou = neighborhood[row_cin][col_fou];

    let d_unodos_two = dos_two - uno_two;
    let d_dostre_two = tre_two - dos_two;
    let d_trequa_two = qua_two - tre_two;
    let d_quacin_two = cin_two - qua_two;
    let d_unodos_thr = dos_thr - uno_thr;
    let d_dostre_thr = tre_thr - dos_thr;
    let d_trequa_thr = qua_thr - tre_thr;
    let d_quacin_thr = cin_thr - qua_thr;
    let d_unodos_fou = dos_fou - uno_fou;
    let d_dostre_fou = tre_fou - dos_fou;
    let d_trequa_fou = qua_fou - tre_fou;
    let d_quacin_fou = cin_fou - qua_fou;
    let d_dos_onetwo = dos_two - dos_one;
    let d_dos_twothr = dos_thr - dos_two;
    let d_dos_thrfou = dos_fou - dos_thr;
    let d_dos_foufiv = dos_fiv - dos_fou;
    let d_tre_onetwo = tre_two - tre_one;
    let d_tre_twothr = tre_thr - tre_two;
    let d_tre_thrfou = tre_fou - tre_thr;
    let d_tre_foufiv = tre_fiv - tre_fou;
    let d_qua_onetwo = qua_two - qua_one;
    let d_qua_twothr = qua_thr - qua_two;
    let d_qua_thrfou = qua_fou - qua_thr;
    let d_qua_foufiv = qua_fiv - qua_fou;

    let d_unodos_times_dostre_two = d_unodos_two * d_dostre_two;
    let d_dostre_two_sq = d_dostre_two * d_dostre_two;
    let d_dostre_times_trequa_two = d_dostre_two * d_trequa_two;
    let d_trequa_times_quacin_two = d_quacin_two * d_trequa_two;
    let d_quacin_two_sq = d_quacin_two * d_quacin_two;
    let d_unodos_times_dostre_thr = d_unodos_thr * d_dostre_thr;
    let d_dostre_thr_sq = d_dostre_thr * d_dostre_thr;
    let d_dostre_times_trequa_thr = d_trequa_thr * d_dostre_thr;
    let d_trequa_times_quacin_thr = d_trequa_thr * d_quacin_thr;
    let d_quacin_thr_sq = d_quacin_thr * d_quacin_thr;
    let d_unodos_times_dostre_fou = d_unodos_fou * d_dostre_fou;
    let d_dostre_fou_sq = d_dostre_fou * d_dostre_fou;
    let d_dostre_times_trequa_fou = d_trequa_fou * d_dostre_fou;
    let d_trequa_times_quacin_fou = d_trequa_fou * d_quacin_fou;
    let d_quacin_fou_sq = d_quacin_fou * d_quacin_fou;
    let d_dos_onetwo_times_twothr = d_dos_onetwo * d_dos_twothr;
    let d_dos_twothr_sq = d_dos_twothr * d_dos_twothr;
    let d_dos_twothr_times_thrfou = d_dos_twothr * d_dos_thrfou;
    let d_dos_thrfou_times_foufiv = d_dos_thrfou * d_dos_foufiv;
    let d_dos_foufiv_sq = d_dos_foufiv * d_dos_foufiv;
    let d_tre_onetwo_times_twothr = d_tre_onetwo * d_tre_twothr;
    let d_tre_twothr_sq = d_tre_twothr * d_tre_twothr;
    let d_tre_twothr_times_thrfou = d_tre_thrfou * d_tre_twothr;
    let d_tre_thrfou_times_foufiv = d_tre_thrfou * d_tre_foufiv;
    let d_tre_foufiv_sq = d_tre_foufiv * d_tre_foufiv;
    let d_qua_onetwo_times_twothr = d_qua_onetwo * d_qua_twothr;
    let d_qua_twothr_sq = d_qua_twothr * d_qua_twothr;
    let d_qua_twothr_times_thrfou = d_qua_thrfou * d_qua_twothr;
    let d_qua_thrfou_times_foufiv = d_qua_thrfou * d_qua_foufiv;
    let d_qua_foufiv_sq = d_qua_foufiv * d_qua_foufiv;

    let dos_thr_y = nohalo_minmod(
        d_dostre_thr,
        d_unodos_thr,
        d_dostre_thr_sq,
        d_unodos_times_dostre_thr,
    );
    let tre_thr_y = nohalo_minmod(
        d_dostre_thr,
        d_trequa_thr,
        d_dostre_thr_sq,
        d_dostre_times_trequa_thr,
    );
    let newval_uno_two = 0.25f64.mul_add(dos_thr_y - tre_thr_y, 0.5 * (dos_thr + tre_thr));

    let qua_thr_y = nohalo_minmod(
        d_quacin_thr,
        d_trequa_thr,
        d_quacin_thr_sq,
        d_trequa_times_quacin_thr,
    );
    let newval_tre_two = 0.25f64.mul_add(tre_thr_y - qua_thr_y, 0.5 * (tre_thr + qua_thr));

    let tre_fou_y = nohalo_minmod(
        d_dostre_fou,
        d_trequa_fou,
        d_dostre_fou_sq,
        d_dostre_times_trequa_fou,
    );
    let qua_fou_y = nohalo_minmod(
        d_quacin_fou,
        d_trequa_fou,
        d_quacin_fou_sq,
        d_trequa_times_quacin_fou,
    );
    let newval_tre_fou = 0.25f64.mul_add(tre_fou_y - qua_fou_y, 0.5 * (tre_fou + qua_fou));

    let dos_fou_y = nohalo_minmod(
        d_dostre_fou,
        d_unodos_fou,
        d_dostre_fou_sq,
        d_unodos_times_dostre_fou,
    );
    let newval_uno_fou = 0.25f64.mul_add(dos_fou_y - tre_fou_y, 0.5 * (dos_fou + tre_fou));

    let tre_two_x = nohalo_minmod(
        d_tre_twothr,
        d_tre_onetwo,
        d_tre_twothr_sq,
        d_tre_onetwo_times_twothr,
    );
    let tre_thr_x = nohalo_minmod(
        d_tre_twothr,
        d_tre_thrfou,
        d_tre_twothr_sq,
        d_tre_twothr_times_thrfou,
    );
    let newval_dos_one = 0.25f64.mul_add(tre_two_x - tre_thr_x, 0.5 * (tre_two + tre_thr));

    let tre_fou_x = nohalo_minmod(
        d_tre_foufiv,
        d_tre_thrfou,
        d_tre_foufiv_sq,
        d_tre_thrfou_times_foufiv,
    );
    let tre_thr_x_minus_tre_fou_x = tre_thr_x - tre_fou_x;
    let newval_dos_thr = 0.25f64.mul_add(tre_thr_x_minus_tre_fou_x, 0.5 * (tre_thr + tre_fou));

    let qua_thr_x = nohalo_minmod(
        d_qua_twothr,
        d_qua_thrfou,
        d_qua_twothr_sq,
        d_qua_twothr_times_thrfou,
    );
    let qua_fou_x = nohalo_minmod(
        d_qua_foufiv,
        d_qua_thrfou,
        d_qua_foufiv_sq,
        d_qua_thrfou_times_foufiv,
    );
    let qua_thr_x_minus_qua_fou_x = qua_thr_x - qua_fou_x;
    let newval_qua_thr = 0.25f64.mul_add(qua_thr_x_minus_qua_fou_x, 0.5 * (qua_thr + qua_fou));

    let qua_two_x = nohalo_minmod(
        d_qua_twothr,
        d_qua_onetwo,
        d_qua_twothr_sq,
        d_qua_onetwo_times_twothr,
    );
    let newval_qua_one = 0.25f64.mul_add(qua_two_x - qua_thr_x, 0.5 * (qua_two + qua_thr));

    let newval_tre_thr = 0.5f64.mul_add(
        newval_tre_two + newval_tre_fou,
        0.125 * (tre_thr_x_minus_tre_fou_x + qua_thr_x_minus_qua_fou_x),
    );

    let dos_thr_x = nohalo_minmod(
        d_dos_twothr,
        d_dos_thrfou,
        d_dos_twothr_sq,
        d_dos_twothr_times_thrfou,
    );
    let dos_fou_x = nohalo_minmod(
        d_dos_foufiv,
        d_dos_thrfou,
        d_dos_foufiv_sq,
        d_dos_thrfou_times_foufiv,
    );
    let newval_uno_thr = 0.5f64.mul_add(
        newval_uno_two + newval_dos_thr,
        0.125f64.mul_add(
            dos_fou_y - tre_fou_y + dos_thr_x - dos_fou_x,
            0.25 * (dos_fou - tre_thr),
        ),
    );

    let tre_two_y = nohalo_minmod(
        d_dostre_two,
        d_trequa_two,
        d_dostre_two_sq,
        d_dostre_times_trequa_two,
    );
    let qua_two_y = nohalo_minmod(
        d_quacin_two,
        d_trequa_two,
        d_quacin_two_sq,
        d_trequa_times_quacin_two,
    );
    let newval_tre_one = 0.5f64.mul_add(
        newval_dos_one + newval_tre_two,
        0.125f64.mul_add(
            qua_two_x - qua_thr_x + tre_two_y - qua_two_y,
            0.25 * (qua_two - tre_thr),
        ),
    );

    let dos_two_x = nohalo_minmod(
        d_dos_twothr,
        d_dos_onetwo,
        d_dos_twothr_sq,
        d_dos_onetwo_times_twothr,
    );
    let dos_two_y = nohalo_minmod(
        d_dostre_two,
        d_unodos_two,
        d_dostre_two_sq,
        d_unodos_times_dostre_two,
    );
    let newval_uno_one = 0.125f64.mul_add(
        dos_two_x - dos_thr_x + tre_two_x - tre_thr_x + dos_two_y + dos_thr_y
            - tre_two_y
            - tre_thr_y,
        0.25 * (dos_two + dos_thr + tre_two + tre_thr),
    );

    [
        newval_uno_one,
        newval_uno_two,
        newval_uno_thr,
        newval_uno_fou,
        newval_dos_one,
        tre_thr,
        newval_dos_thr,
        tre_fou,
        newval_tre_one,
        newval_tre_two,
        newval_tre_thr,
        newval_tre_fou,
        newval_qua_one,
        qua_thr,
        newval_qua_thr,
        qua_fou,
    ]
}

#[inline]
fn nohalo_lbbicubic(stencil: &[f64; 16], relative_x: f64, relative_y: f64) -> f64 {
    let [
        uno_one,
        uno_two,
        uno_thr,
        uno_fou,
        dos_one,
        dos_two,
        dos_thr,
        dos_fou,
        tre_one,
        tre_two,
        tre_thr,
        tre_fou,
        qua_one,
        qua_two,
        qua_thr,
        qua_fou,
    ] = *stencil;

    let m1 = if dos_two <= dos_thr { dos_two } else { dos_thr };
    let max1 = if dos_two <= dos_thr { dos_thr } else { dos_two };
    let m2 = if tre_two <= tre_thr { tre_two } else { tre_thr };
    let max2 = if tre_two <= tre_thr { tre_thr } else { tre_two };
    let m3 = if uno_two <= uno_thr { uno_two } else { uno_thr };
    let max3 = if uno_two <= uno_thr { uno_thr } else { uno_two };
    let m4 = if qua_two <= qua_thr { qua_two } else { qua_thr };
    let max4 = if qua_two <= qua_thr { qua_thr } else { qua_two };
    let m5 = nohalo_min(m1, m2);
    let max5 = nohalo_max(max1, max2);
    let m6 = if dos_one <= tre_one { dos_one } else { tre_one };
    let max6 = if dos_one <= tre_one { tre_one } else { dos_one };
    let m7 = if dos_fou <= tre_fou { dos_fou } else { tre_fou };
    let max7 = if dos_fou <= tre_fou { tre_fou } else { dos_fou };
    let m13 = if dos_fou <= qua_fou { dos_fou } else { qua_fou };
    let max13 = if dos_fou <= qua_fou { qua_fou } else { dos_fou };
    let m9 = nohalo_min(m5, m4);
    let max9 = nohalo_max(max5, max4);
    let m11 = nohalo_min(m6, qua_one);
    let max11 = nohalo_max(max6, qua_one);
    let m10 = nohalo_min(m6, uno_one);
    let max10 = nohalo_max(max6, uno_one);
    let m8 = nohalo_min(m5, m3);
    let max8 = nohalo_max(max5, max3);
    let m12 = nohalo_min(m7, uno_fou);
    let max12 = nohalo_max(max7, uno_fou);
    let min11 = nohalo_min(m9, m13);
    let max11_corner = nohalo_max(max9, max13);
    let min01 = nohalo_min(m9, m11);
    let max01 = nohalo_max(max9, max11);
    let min00 = nohalo_min(m8, m10);
    let max00 = nohalo_max(max8, max10);
    let min10 = nohalo_min(m8, m12);
    let max10_corner = nohalo_max(max8, max12);

    let u11 = tre_thr - min11;
    let v11 = max11_corner - tre_thr;
    let u01 = tre_two - min01;
    let v01 = max01 - tre_two;
    let u00 = dos_two - min00;
    let v00 = max00 - dos_two;
    let u10 = dos_thr - min10;
    let v10 = max10_corner - dos_thr;

    let dble_dzdx00i = dos_thr - dos_one;
    let dble_dzdy11i = qua_thr - dos_thr;
    let dble_dzdx10i = dos_fou - dos_two;
    let dble_dzdy01i = qua_two - dos_two;
    let dble_dzdx01i = tre_thr - tre_one;
    let dble_dzdy10i = tre_thr - uno_thr;
    let dble_dzdx11i = tre_fou - tre_two;
    let dble_dzdy00i = tre_two - uno_two;

    let sign_dzdx00 = if dble_dzdx00i >= 0.0 { 1.0 } else { -1.0 };
    let sign_dzdx10 = if dble_dzdx10i >= 0.0 { 1.0 } else { -1.0 };
    let sign_dzdx01 = if dble_dzdx01i >= 0.0 { 1.0 } else { -1.0 };
    let sign_dzdx11 = if dble_dzdx11i >= 0.0 { 1.0 } else { -1.0 };
    let sign_dzdy00 = if dble_dzdy00i >= 0.0 { 1.0 } else { -1.0 };
    let sign_dzdy10 = if dble_dzdy10i >= 0.0 { 1.0 } else { -1.0 };
    let sign_dzdy01 = if dble_dzdy01i >= 0.0 { 1.0 } else { -1.0 };
    let sign_dzdy11 = if dble_dzdy11i >= 0.0 { 1.0 } else { -1.0 };

    let quad_d2zdxdy00i = uno_one - uno_thr + dble_dzdx01i;
    let quad_d2zdxdy10i = uno_two - uno_fou + dble_dzdx11i;
    let quad_d2zdxdy01i = qua_thr - qua_one - dble_dzdx00i;
    let quad_d2zdxdy11i = qua_fou - qua_two - dble_dzdx10i;

    let dble_slopelimit_00 = 6.0 * nohalo_min(u00, v00);
    let dble_slopelimit_10 = 6.0 * nohalo_min(u10, v10);
    let dble_slopelimit_01 = 6.0 * nohalo_min(u01, v01);
    let dble_slopelimit_11 = 6.0 * nohalo_min(u11, v11);

    let dble_dzdx00 = if sign_dzdx00 * dble_dzdx00i <= dble_slopelimit_00 {
        dble_dzdx00i
    } else {
        sign_dzdx00 * dble_slopelimit_00
    };
    let dble_dzdy00 = if sign_dzdy00 * dble_dzdy00i <= dble_slopelimit_00 {
        dble_dzdy00i
    } else {
        sign_dzdy00 * dble_slopelimit_00
    };
    let dble_dzdx10 = if sign_dzdx10 * dble_dzdx10i <= dble_slopelimit_10 {
        dble_dzdx10i
    } else {
        sign_dzdx10 * dble_slopelimit_10
    };
    let dble_dzdy10 = if sign_dzdy10 * dble_dzdy10i <= dble_slopelimit_10 {
        dble_dzdy10i
    } else {
        sign_dzdy10 * dble_slopelimit_10
    };
    let dble_dzdx01 = if sign_dzdx01 * dble_dzdx01i <= dble_slopelimit_01 {
        dble_dzdx01i
    } else {
        sign_dzdx01 * dble_slopelimit_01
    };
    let dble_dzdy01 = if sign_dzdy01 * dble_dzdy01i <= dble_slopelimit_01 {
        dble_dzdy01i
    } else {
        sign_dzdy01 * dble_slopelimit_01
    };
    let dble_dzdx11 = if sign_dzdx11 * dble_dzdx11i <= dble_slopelimit_11 {
        dble_dzdx11i
    } else {
        sign_dzdx11 * dble_slopelimit_11
    };
    let dble_dzdy11 = if sign_dzdy11 * dble_dzdy11i <= dble_slopelimit_11 {
        dble_dzdy11i
    } else {
        sign_dzdy11 * dble_slopelimit_11
    };

    let twelve_sum00 = 6.0 * (dble_dzdx00 + dble_dzdy00);
    let twelve_dif00 = 6.0 * (dble_dzdx00 - dble_dzdy00);
    let twelve_sum10 = 6.0 * (dble_dzdx10 + dble_dzdy10);
    let twelve_dif10 = 6.0 * (dble_dzdx10 - dble_dzdy10);
    let twelve_sum01 = 6.0 * (dble_dzdx01 + dble_dzdy01);
    let twelve_dif01 = 6.0 * (dble_dzdx01 - dble_dzdy01);
    let twelve_sum11 = 6.0 * (dble_dzdx11 + dble_dzdy11);
    let twelve_dif11 = 6.0 * (dble_dzdx11 - dble_dzdy11);

    let twelve_abs_sum00 = twelve_sum00.abs();
    let twelve_abs_sum10 = twelve_sum10.abs();
    let twelve_abs_sum01 = twelve_sum01.abs();
    let twelve_abs_sum11 = twelve_sum11.abs();

    let u00_times_36 = 36.0 * u00;
    let u10_times_36 = 36.0 * u10;
    let u01_times_36 = 36.0 * u01;
    let u11_times_36 = 36.0 * u11;

    let first_limit00 = twelve_abs_sum00 - u00_times_36;
    let first_limit10 = twelve_abs_sum10 - u10_times_36;
    let first_limit01 = twelve_abs_sum01 - u01_times_36;
    let first_limit11 = twelve_abs_sum11 - u11_times_36;

    let quad_d2zdxdy00ii = nohalo_max(quad_d2zdxdy00i, first_limit00);
    let quad_d2zdxdy10ii = nohalo_max(quad_d2zdxdy10i, first_limit10);
    let quad_d2zdxdy01ii = nohalo_max(quad_d2zdxdy01i, first_limit01);
    let quad_d2zdxdy11ii = nohalo_max(quad_d2zdxdy11i, first_limit11);

    let v00_times_36 = 36.0 * v00;
    let v10_times_36 = 36.0 * v10;
    let v01_times_36 = 36.0 * v01;
    let v11_times_36 = 36.0 * v11;

    let second_limit00 = v00_times_36 - twelve_abs_sum00;
    let second_limit10 = v10_times_36 - twelve_abs_sum10;
    let second_limit01 = v01_times_36 - twelve_abs_sum01;
    let second_limit11 = v11_times_36 - twelve_abs_sum11;

    let quad_d2zdxdy00iii = nohalo_min(quad_d2zdxdy00ii, second_limit00);
    let quad_d2zdxdy10iii = nohalo_min(quad_d2zdxdy10ii, second_limit10);
    let quad_d2zdxdy01iii = nohalo_min(quad_d2zdxdy01ii, second_limit01);
    let quad_d2zdxdy11iii = nohalo_min(quad_d2zdxdy11ii, second_limit11);

    let twelve_abs_dif00 = twelve_dif00.abs();
    let twelve_abs_dif10 = twelve_dif10.abs();
    let twelve_abs_dif01 = twelve_dif01.abs();
    let twelve_abs_dif11 = twelve_dif11.abs();

    let third_limit00 = twelve_abs_dif00 - v00_times_36;
    let third_limit10 = twelve_abs_dif10 - v10_times_36;
    let third_limit01 = twelve_abs_dif01 - v01_times_36;
    let third_limit11 = twelve_abs_dif11 - v11_times_36;

    let quad_d2zdxdy00iiii = nohalo_max(quad_d2zdxdy00iii, third_limit00);
    let quad_d2zdxdy10iiii = nohalo_max(quad_d2zdxdy10iii, third_limit10);
    let quad_d2zdxdy01iiii = nohalo_max(quad_d2zdxdy01iii, third_limit01);
    let quad_d2zdxdy11iiii = nohalo_max(quad_d2zdxdy11iii, third_limit11);

    let fourth_limit00 = u00_times_36 - twelve_abs_dif00;
    let fourth_limit10 = u10_times_36 - twelve_abs_dif10;
    let fourth_limit01 = u01_times_36 - twelve_abs_dif01;
    let fourth_limit11 = u11_times_36 - twelve_abs_dif11;

    let quad_d2zdxdy00 = nohalo_min(quad_d2zdxdy00iiii, fourth_limit00);
    let quad_d2zdxdy10 = nohalo_min(quad_d2zdxdy10iiii, fourth_limit10);
    let quad_d2zdxdy01 = nohalo_min(quad_d2zdxdy01iiii, fourth_limit01);
    let quad_d2zdxdy11 = nohalo_min(quad_d2zdxdy11iiii, fourth_limit11);

    let xp1over2 = relative_x;
    let xm1over2 = xp1over2 - 1.0;
    let onepx = 0.5 + xp1over2;
    let onemx = 1.5 - xp1over2;
    let xp1over2sq = xp1over2 * xp1over2;

    let yp1over2 = relative_y;
    let ym1over2 = yp1over2 - 1.0;
    let onepy = 0.5 + yp1over2;
    let onemy = 1.5 - yp1over2;
    let yp1over2sq = yp1over2 * yp1over2;

    let xm1over2sq = xm1over2 * xm1over2;
    let ym1over2sq = ym1over2 * ym1over2;

    let twice1px = onepx + onepx;
    let twice1py = onepy + onepy;
    let twice1mx = onemx + onemx;
    let twice1my = onemy + onemy;

    let xm1over2sq_times_ym1over2sq = xm1over2sq * ym1over2sq;
    let xp1over2sq_times_ym1over2sq = xp1over2sq * ym1over2sq;
    let xp1over2sq_times_yp1over2sq = xp1over2sq * yp1over2sq;
    let xm1over2sq_times_yp1over2sq = xm1over2sq * yp1over2sq;

    let four_times_1px_times_1py = twice1px * twice1py;
    let four_times_1mx_times_1py = twice1mx * twice1py;
    let twice_xp1over2_times_1py = xp1over2 * twice1py;
    let twice_xm1over2_times_1py = xm1over2 * twice1py;
    let twice_xm1over2_times_1my = xm1over2 * twice1my;
    let twice_xp1over2_times_1my = xp1over2 * twice1my;
    let four_times_1mx_times_1my = twice1mx * twice1my;
    let four_times_1px_times_1my = twice1px * twice1my;
    let twice_1px_times_ym1over2 = twice1px * ym1over2;
    let twice_1mx_times_ym1over2 = twice1mx * ym1over2;
    let xp1over2_times_ym1over2 = xp1over2 * ym1over2;
    let xm1over2_times_ym1over2 = xm1over2 * ym1over2;
    let xm1over2_times_yp1over2 = xm1over2 * yp1over2;
    let xp1over2_times_yp1over2 = xp1over2 * yp1over2;
    let twice_1mx_times_yp1over2 = twice1mx * yp1over2;
    let twice_1px_times_yp1over2 = twice1px * yp1over2;

    let c00 = four_times_1px_times_1py * xm1over2sq_times_ym1over2sq;
    let c00dx = twice_xp1over2_times_1py * xm1over2sq_times_ym1over2sq;
    let c00dy = twice_1px_times_yp1over2 * xm1over2sq_times_ym1over2sq;
    let c00dxdy = xp1over2_times_yp1over2 * xm1over2sq_times_ym1over2sq;
    let c10 = four_times_1mx_times_1py * xp1over2sq_times_ym1over2sq;
    let c10dx = twice_xm1over2_times_1py * xp1over2sq_times_ym1over2sq;
    let c10dy = twice_1mx_times_yp1over2 * xp1over2sq_times_ym1over2sq;
    let c10dxdy = xm1over2_times_yp1over2 * xp1over2sq_times_ym1over2sq;
    let c01 = four_times_1px_times_1my * xm1over2sq_times_yp1over2sq;
    let c01dx = twice_xp1over2_times_1my * xm1over2sq_times_yp1over2sq;
    let c01dy = twice_1px_times_ym1over2 * xm1over2sq_times_yp1over2sq;
    let c01dxdy = xp1over2_times_ym1over2 * xm1over2sq_times_yp1over2sq;
    let c11 = four_times_1mx_times_1my * xp1over2sq_times_yp1over2sq;
    let c11dx = twice_xm1over2_times_1my * xp1over2sq_times_yp1over2sq;
    let c11dy = twice_1mx_times_ym1over2 * xp1over2sq_times_yp1over2sq;
    let c11dxdy = xm1over2_times_ym1over2 * xp1over2sq_times_yp1over2sq;

    let newval1 = c00 * dos_two + c10 * dos_thr + c01 * tre_two + c11 * tre_thr;
    let newval2 = c00dx * dble_dzdx00
        + c10dx * dble_dzdx10
        + c01dx * dble_dzdx01
        + c11dx * dble_dzdx11
        + c00dy * dble_dzdy00
        + c10dy * dble_dzdy10
        + c01dy * dble_dzdy01
        + c11dy * dble_dzdy11;
    let newval3 = c00dxdy * quad_d2zdxdy00
        + c10dxdy * quad_d2zdxdy10
        + c01dxdy * quad_d2zdxdy01
        + c11dxdy * quad_d2zdxdy11;

    0.25f64.mul_add(newval3, 0.5f64.mul_add(newval2, newval1))
}

#[inline]
fn vsqbs_anchor_and_sign(phase: f64) -> (usize, i32, f64) {
    debug_assert!((0.0..1.0).contains(&phase), "phase must be in [0, 1)");

    if phase < 0.5 {
        (1, 1, phase)
    } else {
        (2, -1, phase - 1.0)
    }
}

#[inline]
fn interpolate_vsqbs_2d<const N: usize>(
    x_phase: f64,
    y_phase: f64,
    neighborhood: &[[f64; N]; N],
) -> f64 {
    debug_assert!(N >= 4, "vsqbs requires a 4x4 neighborhood");

    let (anchor_x, sign_x, x_0) = vsqbs_anchor_and_sign(x_phase);
    let (anchor_y, sign_y, y_0) = vsqbs_anchor_and_sign(y_phase);

    let twice_abs_x_0 = f64::from(2 * sign_x) * x_0;
    let twice_abs_y_0 = f64::from(2 * sign_y) * y_0;
    let x = twice_abs_x_0 - 0.5;
    let y = twice_abs_y_0 - 0.5;
    let cent = 0.75 - x * x;
    let mid = 0.75 - y * y;
    let left = (-0.5f64).mul_add(x + cent, 0.5);
    let top = (-0.5f64).mul_add(y + mid, 0.5);
    let left_p_cent = left + cent;
    let top_p_mid = top + mid;
    let cent_p_rite = 1.0 - left;
    let mid_p_bot = 1.0 - top;
    let rite = 1.0 - left_p_cent;
    let bot = 1.0 - top_p_mid;

    let four_c_uno_two = left_p_cent * top;
    let four_c_dos_one = left * top_p_mid;
    let four_c_dos_two = left_p_cent + top_p_mid;
    let four_c_dos_thr = cent_p_rite * top_p_mid + rite;
    let four_c_tre_two = mid_p_bot * left_p_cent + bot;
    let four_c_tre_thr = mid_p_bot * rite + cent_p_rite * bot;
    let four_c_uno_thr = top - four_c_uno_two;
    let four_c_tre_one = left - four_c_dos_one;

    let row_top = (anchor_y as i32 - sign_y) as usize;
    let row_mid = anchor_y;
    let row_bot = (anchor_y as i32 + sign_y) as usize;
    let col_left = (anchor_x as i32 - sign_x) as usize;
    let col_mid = anchor_x;
    let col_right = (anchor_x as i32 + sign_x) as usize;

    (((four_c_uno_two * neighborhood[row_top][col_mid]
        + four_c_dos_one * neighborhood[row_mid][col_left])
        + (four_c_dos_two * neighborhood[row_mid][col_mid]
            + four_c_dos_thr * neighborhood[row_mid][col_right]))
        + ((four_c_tre_two * neighborhood[row_bot][col_mid]
            + four_c_tre_thr * neighborhood[row_bot][col_right])
            + (four_c_uno_thr * neighborhood[row_top][col_right]
                + four_c_tre_one * neighborhood[row_bot][col_left])))
        * 0.25
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    const EPSILON: f64 = 1e-12;
    const NOHALO_SUPPORT: f64 = 2.5;

    fn nohalo_approximate_weight(x: f64) -> f64 {
        let x = x.abs();
        if x < f64::EPSILON {
            1.0
        } else if x < NOHALO_SUPPORT {
            let pi_x = std::f64::consts::PI * x;
            let sinc = pi_x.sin() / pi_x;
            let window = 0.5 * (1.0 + (pi_x / NOHALO_SUPPORT).cos());
            sinc * window
        } else {
            0.0
        }
    }

    fn implemented_kernels() -> [InterpolationKernel; 9] {
        [
            InterpolationKernel::Nearest,
            InterpolationKernel::Bilinear,
            InterpolationKernel::Bicubic,
            InterpolationKernel::Quadratic,
            InterpolationKernel::CatmullRom,
            InterpolationKernel::Nohalo,
            InterpolationKernel::Lbb,
            InterpolationKernel::Lanczos2,
            InterpolationKernel::Lanczos3,
        ]
    }

    fn partition_kernels() -> [InterpolationKernel; 5] {
        [
            InterpolationKernel::Nearest,
            InterpolationKernel::Bilinear,
            InterpolationKernel::Bicubic,
            InterpolationKernel::Quadratic,
            InterpolationKernel::CatmullRom,
        ]
    }

    fn partition_of_unity(kernel: InterpolationKernel, phase: f64) -> f64 {
        let support = kernel.support().ceil() as i32;
        let mut sum = 0.0;
        for tap in -support..=support {
            let sample = tap as f64;
            sum += kernel.interpolate((phase - sample).abs());
        }
        sum
    }

    #[test]
    fn nearest_support_and_weight() {
        assert_eq!(InterpolationKernel::Nearest.support(), 0.5);
        assert_eq!(InterpolationKernel::Nearest.interpolate(0.0), 1.0);
        assert_eq!(InterpolationKernel::Nearest.interpolate(0.5), 0.0);
    }

    #[test]
    fn implemented_kernels_interpolate_to_one_at_origin() {
        for kernel in implemented_kernels() {
            assert!(
                (kernel.interpolate(0.0) - 1.0).abs() < EPSILON,
                "kernel={kernel:?}"
            );
        }
    }

    #[test]
    fn bicubic_matches_catmull_rom_formula() {
        for sample in [0.0, 0.25, 0.5, 1.0, 1.5, 2.0] {
            assert!(
                (InterpolationKernel::Bicubic.interpolate(sample)
                    - InterpolationKernel::CatmullRom.interpolate(sample))
                .abs()
                    < EPSILON
            );
        }
    }

    #[test]
    fn quadratic_support_and_origin_weight_match_reference() {
        assert_eq!(InterpolationKernel::Quadratic.support(), 1.5);
        assert!((InterpolationKernel::Quadratic.interpolate(0.0) - 1.0).abs() < EPSILON);
    }

    #[test]
    fn quadratic_zeroes_at_and_beyond_support() {
        for sample in [1.5, 1.75, 3.0] {
            assert!(InterpolationKernel::Quadratic.interpolate(sample).abs() < EPSILON);
            assert!(InterpolationKernel::Quadratic.interpolate(-sample).abs() < EPSILON);
        }
    }

    #[test]
    fn bilinear_zero_at_support() {
        assert!(InterpolationKernel::Bilinear.interpolate(1.0).abs() < EPSILON);
    }

    #[test]
    fn halo_for_factor_nearest() {
        assert_eq!(InterpolationKernel::Nearest.halo_for_factor(2.0), 1);
    }

    #[test]
    fn halo_for_factor_lanczos3_factor1() {
        assert_eq!(InterpolationKernel::Lanczos3.halo_for_factor(1.0), 3);
    }

    #[test]
    fn halo_for_factor_bicubic_factor2() {
        assert_eq!(InterpolationKernel::Bicubic.halo_for_factor(2.0), 1);
    }

    #[test]
    fn nohalo_uses_six_tap_affine_window() {
        assert!((InterpolationKernel::Nohalo.support() - NOHALO_SUPPORT).abs() < EPSILON);
        assert_eq!(InterpolationKernel::Nohalo.window_size(), 6);
        assert_eq!(InterpolationKernel::Nohalo.window_offset(), 2);
        assert_eq!(InterpolationKernel::Nohalo.affine_padding(), (2, 3));
        assert_eq!(InterpolationKernel::Nohalo.halo_for_factor(1.0), 3);
    }

    #[test]
    fn nohalo_scalar_fallback_matches_windowed_sinc_approximation_samples() {
        for sample in [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 2.499, 2.5, 3.0] {
            let expected = nohalo_approximate_weight(sample);
            let actual = InterpolationKernel::Nohalo.interpolate(sample);
            assert!(
                (actual - expected).abs() < 1e-12,
                "sample={sample} actual={actual} expected={expected}"
            );
        }
    }

    #[test]
    fn lbb_and_vsqbs_use_four_tap_affine_window() {
        assert_eq!(InterpolationKernel::Lbb.affine_padding(), (1, 2));
        assert_eq!(InterpolationKernel::Vsqbs.affine_padding(), (1, 2));
    }

    #[test]
    fn nohalo_2d_preserves_constant_neighborhood() {
        let neighborhood = [[7.0; 6]; 6];

        for (x_phase, y_phase) in [(0.0, 0.0), (0.25, 0.25), (0.5, 0.5), (0.75, 0.75)] {
            assert!(
                (InterpolationKernel::Nohalo.interpolate_2d(x_phase, y_phase, &neighborhood) - 7.0)
                    .abs()
                    < EPSILON,
                "x_phase={x_phase} y_phase={y_phase}",
            );
        }
    }

    #[test]
    fn vsqbs_2d_preserves_constant_neighborhood() {
        let neighborhood = [[7.0; 4]; 4];

        for (x_phase, y_phase) in [(0.0, 0.0), (0.25, 0.25), (0.25, 0.75), (0.75, 0.75)] {
            assert!(
                (InterpolationKernel::Vsqbs.interpolate_2d(x_phase, y_phase, &neighborhood) - 7.0)
                    .abs()
                    < EPSILON,
                "x_phase={x_phase} y_phase={y_phase}",
            );
        }
    }

    #[test]
    fn vsqbs_2d_matches_reference_stencil_value() {
        let neighborhood = [
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0, 7.0],
            [8.0, 9.0, 10.0, 11.0],
            [12.0, 13.0, 14.0, 15.0],
        ];

        assert!(
            (InterpolationKernel::Vsqbs.interpolate_2d(0.25, 0.75, &neighborhood) - 8.25).abs()
                < EPSILON
        );
    }

    #[test]
    fn lbb_interpolate_is_one_at_origin() {
        assert!((InterpolationKernel::Lbb.interpolate(0.0) - 1.0).abs() < EPSILON);
    }

    proptest! {
        #[test]
        fn implemented_kernels_form_partition_of_unity(phase in 0.0f64..1.0) {
            for kernel in partition_kernels() {
                let sum = partition_of_unity(kernel, phase);
                prop_assert!((sum - 1.0).abs() < 1e-9, "kernel={kernel:?} phase={phase} sum={sum}");
            }
        }

        #[test]
        fn nohalo_2d_stays_within_neighborhood_bounds(
            samples in prop::collection::vec(-32.0f64..32.0, 36),
            x_phase in 0.0f64..1.0,
            y_phase in 0.0f64..1.0,
        ) {
            let neighborhood = [
                [samples[0], samples[1], samples[2], samples[3], samples[4], samples[5]],
                [samples[6], samples[7], samples[8], samples[9], samples[10], samples[11]],
                [samples[12], samples[13], samples[14], samples[15], samples[16], samples[17]],
                [samples[18], samples[19], samples[20], samples[21], samples[22], samples[23]],
                [samples[24], samples[25], samples[26], samples[27], samples[28], samples[29]],
                [samples[30], samples[31], samples[32], samples[33], samples[34], samples[35]],
            ];
            let value = InterpolationKernel::Nohalo.interpolate_2d(x_phase, y_phase, &neighborhood);
            let min = samples.iter().copied().fold(f64::INFINITY, f64::min);
            let max = samples.iter().copied().fold(f64::NEG_INFINITY, f64::max);

            prop_assert!(value >= min - 1e-9, "value={value} min={min}");
            prop_assert!(value <= max + 1e-9, "value={value} max={max}");
        }

        #[test]
        fn nearest_weight_is_bounded(x in -4.0f64..4.0) {
            let weight = InterpolationKernel::Nearest.interpolate(x);
            prop_assert!(weight.abs() <= 1.0 + 1e-12);
        }

        #[test]
        fn bilinear_weight_is_bounded(x in -4.0f64..4.0) {
            let weight = InterpolationKernel::Bilinear.interpolate(x);
            prop_assert!(weight.abs() <= 1.0 + 1e-12);
        }

        #[test]
        fn quadratic_partition_of_unity(phase in -0.5f64..0.5) {
            let kernel = InterpolationKernel::Quadratic;
            let sum = kernel.interpolate(phase)
                + kernel.interpolate(phase - 1.0)
                + kernel.interpolate(phase + 1.0);
            prop_assert!((sum - 1.0).abs() < 1e-9, "phase={phase} sum={sum}");
        }

        #[test]
        fn quadratic_weight_is_bounded(x in -1.5f64..1.5) {
            let weight = InterpolationKernel::Quadratic.interpolate(x);
            prop_assert!(weight.abs() <= 1.0 + 1e-12, "x={x} weight={weight}");
        }

        #[test]
        fn quadratic_weight_is_zero_outside_support(x in 1.5f64..4.0) {
            prop_assert!(InterpolationKernel::Quadratic.interpolate(x).abs() < EPSILON);
            prop_assert!(InterpolationKernel::Quadratic.interpolate(-x).abs() < EPSILON);
        }

        #[test]
        fn halo_is_at_least_one_for_large_factor(factor in 1.0f64..=32.0) {
            let h = InterpolationKernel::Bicubic.halo_for_factor(factor);
            prop_assert!(h >= 1);
        }
    }
}
