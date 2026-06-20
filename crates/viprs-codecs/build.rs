//! Build script: verifies system libraries are present at compile time.
//!
//! Features that require an external system library will fail compilation
//! with a clear error message if the library is not installed, instead of
//! producing confusing runtime failures.
#![allow(missing_docs, clippy::panic)]

/// Checks that a system library is discoverable via pkg-config.
/// Panics with a user-friendly message if not found.
fn require_system_lib(pkg_name: &str, feature: &str, install_hint: &str) {
    let result = std::process::Command::new("pkg-config")
        .args(["--exists", pkg_name])
        .status();
    match result {
        Ok(status) if status.success() => {}
        _ => {
            panic!(
                "\n\n\
                 ╔══════════════════════════════════════════════════════════════╗\n\
                 ║  COMPILE ERROR: missing system library                      ║\n\
                 ╠══════════════════════════════════════════════════════════════╣\n\
                 ║  Feature `{feature}` requires `{pkg_name}` to be installed.\n\
                 ║\n\
                 ║  Install it:\n\
                 {install_hint}\
                 ║\n\
                 ║  Or compile without this feature.\n\
                 ╚══════════════════════════════════════════════════════════════╝\n\n",
            );
        }
    }
}

fn main() {
    #[cfg(feature = "jpeg")]
    require_system_lib(
        "libturbojpeg",
        "jpeg",
        "\
         ║    macOS:  brew install jpeg-turbo\n\
         ║    Ubuntu: apt install libturbojpeg0-dev\n\
         ║    Fedora: dnf install turbojpeg-devel\n",
    );

    #[cfg(feature = "libspng")]
    require_system_lib(
        "spng",
        "libspng",
        "\
         ║    macOS:  brew install libspng\n\
         ║    Ubuntu: apt install libspng-dev\n\
         ║    Fedora: dnf install libspng-devel\n",
    );

    #[cfg(feature = "heif")]
    require_system_lib(
        "libheif",
        "heif",
        "\
         ║    macOS:  brew install libheif\n\
         ║    Ubuntu: apt install libheif-dev\n\
         ║    Fedora: dnf install libheif-devel\n",
    );

    #[cfg(feature = "jxl")]
    require_system_lib(
        "libjxl",
        "jxl",
        "\
         ║    macOS:  brew install jpeg-xl\n\
         ║    Ubuntu: apt install libjxl-dev\n\
         ║    Fedora: dnf install libjxl-devel\n",
    );

    #[cfg(feature = "icc")]
    require_system_lib(
        "lcms2",
        "icc",
        "\
         ║    macOS:  brew install little-cms2\n\
         ║    Ubuntu: apt install liblcms2-dev\n\
         ║    Fedora: dnf install lcms2-devel\n",
    );

    #[cfg(feature = "openslide")]
    require_system_lib(
        "openslide",
        "openslide",
        "\
         ║    macOS:  brew install openslide\n\
         ║    Ubuntu: apt install libopenslide-dev\n\
         ║    Fedora: dnf install openslide-devel\n",
    );
}
