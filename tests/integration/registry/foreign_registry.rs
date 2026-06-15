#[cfg(feature = "png")]
#[test]
fn image_load_and_save_use_default_foreign_registry() {
    use std::{fs, path::PathBuf};

    use viprs::{Image, U8};

    fn test_output_path() -> PathBuf {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("foreign-registry-tests");
        fs::create_dir_all(&dir).unwrap();
        dir.join(format!("roundtrip-{}.png", std::process::id()))
    }

    let path = test_output_path();
    let original =
        Image::<U8>::from_buffer(2, 2, 3, vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 12, 34, 56])
            .unwrap();

    original.save(&path).unwrap();
    let decoded = Image::<U8>::load(&path).unwrap();

    assert_eq!(decoded.width(), original.width());
    assert_eq!(decoded.height(), original.height());
    assert_eq!(decoded.bands(), original.bands());
    assert_eq!(decoded.pixels(), original.pixels());
}
