use super::{icc_error, lcms_error};
use crate::domain::error::ViprsError;
use lcms2::{Profile, ToneCurve, white_point_from_temp};
use std::{borrow::Cow, fs, path::Path};

const D65_TEMPERATURE_K: f64 = 6504.0;
const ADOBE_RGB_1998_PROFILE_BYTES: &[u8; 568] = include_bytes!("../icc_profiles/adobergb1998.icc");
const PROPHOTO_RGB_PROFILE_BYTES: &[u8; 568] = include_bytes!("../icc_profiles/prophoto.icc");

fn d65_white_point() -> Result<lcms2::CIExyY, ViprsError> {
    white_point_from_temp(D65_TEMPERATURE_K)
        .ok_or_else(|| icc_error("unable to compute D65 white point"))
}

pub(super) fn gray_profile_bytes() -> Result<Vec<u8>, ViprsError> {
    let white_point = d65_white_point()?;
    let gamma = ToneCurve::new(2.2);
    Profile::new_gray(&white_point, &gamma)
        .map_err(lcms_error)?
        .icc()
        .map_err(lcms_error)
}

pub(super) fn lab_profile_bytes() -> Result<Vec<u8>, ViprsError> {
    let white_point = d65_white_point()?;
    Profile::new_lab4_context(lcms2::GlobalContext::new(), &white_point)
        .map_err(lcms_error)?
        .icc()
        .map_err(lcms_error)
}

pub(super) fn xyz_profile_bytes() -> Result<Vec<u8>, ViprsError> {
    Profile::new_xyz().icc().map_err(lcms_error)
}

fn embedded_profile_bytes(bytes: &'static [u8], role: &str) -> Result<Vec<u8>, ViprsError> {
    let _ = open_profile(bytes, role)?;
    Ok(bytes.to_vec())
}

pub(super) fn open_profile(data: &[u8], role: &str) -> Result<Profile, ViprsError> {
    if data.is_empty() {
        return Err(icc_error(format!("{role} profile is empty")));
    }
    Profile::new_icc(data).map_err(lcms_error)
}

fn normalize_profile_name(name: &str) -> Cow<'_, str> {
    Cow::Owned(name.trim().to_ascii_lowercase())
}

fn system_icc_dirs() -> Vec<std::path::PathBuf> {
    let mut dirs: Vec<std::path::PathBuf> = Vec::new();

    #[cfg(target_os = "macos")]
    {
        dirs.push("/Library/ColorSync/Profiles".into());
        dirs.push("/System/Library/ColorSync/Profiles".into());
        if let Some(home) = dirs_home() {
            dirs.push(home.join("Library/ColorSync/Profiles"));
        }
    }

    #[cfg(target_os = "windows")]
    {
        dirs.push(r"C:\Windows\System32\spool\drivers\color".into());
    }

    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        if let Ok(xdg) = std::env::var("XDG_DATA_HOME") {
            dirs.push(std::path::PathBuf::from(xdg).join("color/icc"));
        }
        if let Some(home) = dirs_home() {
            dirs.push(home.join(".local/share/color/icc"));
            dirs.push(home.join(".color/icc"));
        }
        dirs.push("/usr/share/color/icc".into());
        dirs.push("/usr/local/share/color/icc".into());
    }

    dirs
}

fn dirs_home() -> Option<std::path::PathBuf> {
    std::env::var("HOME").ok().map(std::path::PathBuf::from)
}

fn cmyk_filenames() -> &'static [&'static str] {
    &[
        "GenericCMYKProfile.icc",
        "Generic CMYK Profile.icc",
        "USWebCoatedSWOP.icc",
    ]
}

fn p3_filenames() -> &'static [&'static str] {
    &[
        "Display P3.icc",
        "DisplayP3.icc",
        "DCI(P3) RGB.icc",
        "P3D65.icc",
        "P3-D65.icc",
    ]
}

fn search_system_dirs(name: &str, candidates: &[&str]) -> Option<std::path::PathBuf> {
    let dirs = system_icc_dirs();
    for dir in &dirs {
        for candidate in candidates {
            let path = dir.join(candidate);
            if path.exists() {
                return Some(path);
            }
        }
        for ext in &["icc", "ICC", "icm", "ICM"] {
            let path = dir.join(format!("{name}.{ext}"));
            if path.exists() {
                return Some(path);
            }
        }
    }
    None
}

fn load_and_validate_path(path: &Path, label: &str) -> Result<Vec<u8>, ViprsError> {
    let bytes = fs::read(path).map_err(|e| {
        icc_error(format!(
            "failed to read profile from \"{}\": {e}",
            path.display()
        ))
    })?;
    let _ = open_profile(&bytes, label)?;
    Ok(bytes)
}

pub fn profile_load(name: &str) -> Result<Vec<u8>, ViprsError> {
    let normalized = normalize_profile_name(name);
    match normalized.as_ref() {
        "none" => Err(icc_error(
            "profile \"none\" is a sentinel for strip-profile; callers must handle it before calling profile_load",
        )),
        "srgb" => Profile::new_srgb().icc().map_err(lcms_error),
        "lab" => lab_profile_bytes(),
        "xyz" => xyz_profile_bytes(),
        "sgrey" | "gray" | "grey" => gray_profile_bytes(),
        "adobergb" | "adobergb1998" | "adobe-rgb" | "adobe-rgb-1998" => {
            embedded_profile_bytes(ADOBE_RGB_1998_PROFILE_BYTES, "Adobe RGB (1998)")
        }
        "prophoto" | "prophotorgb" | "prophoto-rgb" | "rommrgb" | "romm-rgb" => {
            embedded_profile_bytes(PROPHOTO_RGB_PROFILE_BYTES, "ProPhoto RGB")
        }
        "cmyk" => {
            let candidates = cmyk_filenames();
            if let Some(path) = search_system_dirs("cmyk", candidates) {
                load_and_validate_path(&path, "cmyk")
            } else {
                Err(icc_error(
                    "profile \"cmyk\" not found in system ICC directories; install a CMYK profile (e.g. from a colour management package) or supply an explicit path",
                ))
            }
        }
        "p3" => {
            let candidates = p3_filenames();
            if let Some(path) = search_system_dirs("p3", candidates) {
                load_and_validate_path(&path, "p3")
            } else {
                Err(icc_error(
                    "profile \"p3\" not found in system ICC directories; install a Display P3 profile (e.g. from a colour management package) or supply an explicit path",
                ))
            }
        }
        _ if Path::new(name).is_absolute() || Path::new(name).exists() => {
            load_and_validate_path(Path::new(name), "loaded")
        }
        _ => {
            if let Some(path) = search_system_dirs(name, &[]) {
                load_and_validate_path(&path, name)
            } else {
                Err(icc_error(format!(
                    "profile \"{name}\" not found: not a known alias, not a valid path, and not found in system ICC directories"
                )))
            }
        }
    }
}
