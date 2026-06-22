use super::*;

#[test]
fn band_format_id_is_copy() {
    let id = BandFormatId::U8;
    let id2 = id;
    let id3 = id;
    assert_eq!(id, id2);
    assert_eq!(id, id3);
}

#[test]
fn each_type_has_correct_id() {
    assert_eq!(U8::ID, BandFormatId::U8);
    assert_eq!(U16::ID, BandFormatId::U16);
    assert_eq!(I16::ID, BandFormatId::I16);
    assert_eq!(U32::ID, BandFormatId::U32);
    assert_eq!(I32::ID, BandFormatId::I32);
    assert_eq!(F32::ID, BandFormatId::F32);
    assert_eq!(F64::ID, BandFormatId::F64);
}

#[test]
fn integer_saturating_div_by_zero() {
    assert_eq!(100u8.s_saturating_div(0), u8::MAX);
    assert_eq!(100u16.s_saturating_div(0), u16::MAX);
    assert_eq!(100i16.s_saturating_div(0), i16::MAX);
    assert_eq!(i16::MIN.s_saturating_div(-1), i16::MAX);
}

#[test]
fn float_div_zero_is_ieee() {
    let result = 1.0f32.s_div(0.0f32);
    assert_eq!(result, 0.0);
}

#[test]
fn float_sample_methods() {
    assert!((1.0f32.s_exp().s_ln() - 1.0).abs() < 1e-6);
    assert_eq!(0.0f32.s_ln(), 0.0);
    assert_eq!(0.0f32.s_log10(), 0.0);
    assert!((0.5f32.s_exp10() - 10.0f32.sqrt()).abs() < 1e-6);
    assert_eq!(FloatSample::s_sign(-0.0f32), 0.0);
    assert!((0.0f32.s_floor() - 0.0).abs() < f32::EPSILON);
    assert!((90.0f32.s_sin() - 1.0).abs() < 1e-6);
    assert!((60.0f32.s_cos() - 0.5).abs() < 1e-6);
    assert!((45.0f32.s_tan() - 1.0).abs() < 1e-6);
    assert!((1.0f32.s_asin() - 90.0).abs() < 1e-6);
    assert!((0.0f32.s_acos() - 90.0).abs() < 1e-6);
    assert!((1.0f32.s_atan() - 45.0).abs() < 1e-6);
}

#[test]
fn remainder_matches_libvips_zero_rules() {
    assert_eq!(5u8.s_remainder(0), u8::MAX);
    assert_eq!(5i32.s_remainder(0), -1);
    assert_eq!(5.0f32.s_remainder(0.0), -1.0);
    assert_eq!(i32::MIN.s_remainder(-1), 0);
}

#[test]
fn bitwise_sample_trait_bounds() {
    fn requires_bitwise<T: BitwiseSample>(_: T) {}

    requires_bitwise(0u8);
    requires_bitwise(0u16);
    requires_bitwise(0u32);
}

#[test]
fn saturating_arithmetic_matches_libvips_style_clip() {
    assert_eq!(250u8.s_add(10), u8::MAX);
    assert_eq!(5u8.s_sub(10), 0);
    assert_eq!(30u8.s_mul(10), u8::MAX);

    assert_eq!(i16::MAX.s_add(1), i16::MAX);
    assert_eq!(i16::MIN.s_sub(1), i16::MIN);
    assert_eq!(i16::MAX.s_mul(2), i16::MAX);
    assert_eq!(i16::MIN.s_mul(2), i16::MIN);
}
