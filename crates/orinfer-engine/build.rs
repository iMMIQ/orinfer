use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=native/ascend/hccl_graph_registry.cc");
    println!("cargo:rerun-if-changed=native/ascend/hccl_graph_registry_abi.h");
    println!("cargo:rerun-if-env-changed=ORINFER_STRICT_GLIBC_FLOOR");
    println!("cargo:rerun-if-env-changed=ORINFER_SKIP_GLIBC_FLOOR");
    let output = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("liborinfer_hccl_registry.so");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        let compiler = cc::Build::new().cpp(true).get_compiler();
        let mut command = compiler.to_command();
        command.args([
            "-shared",
            "-fPIC",
            "-O2",
            "-std=c++17",
            "-D_GLIBCXX_USE_CXX11_ABI=0",
            "-fvisibility=hidden",
            "-static-libstdc++",
        ]);
        if compiler.is_like_gnu() {
            command.arg("-shared-libgcc");
        }
        let status = command
            .args([
                "-Wl,--exclude-libs,ALL",
                "-Wl,-z,defs",
                "-Wall",
                "-Wextra",
                "-Werror",
                "native/ascend/hccl_graph_registry.cc",
                "-ldl",
                "-lpthread",
                "-o",
            ])
            .arg(&output)
            .status()
            .expect("C++ compiler for the CANN 9.2 Graph registry adapter");
        assert!(status.success(), "CANN Graph registry adapter build failed");
        check_glibc_floor(&output);
    } else {
        std::fs::write(&output, []).unwrap();
    }
    println!("cargo:rustc-env=ORINFER_HCCL_REGISTRY={}", output.display());
}

fn host_glibc_version() -> Option<(u32, u32)> {
    let output = std::process::Command::new("getconf")
        .arg("GNU_LIBC_VERSION")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    let version = text.split_ascii_whitespace().last()?;
    let mut parts = version.split('.').map(str::parse::<u32>);
    match (parts.next(), parts.next()) {
        (Some(Ok(major)), Some(Ok(minor))) => Some((major, minor)),
        _ => None,
    }
}

// A nested shared object has its own version requirements, independent of the
// Rust executable. Reject an incompatible embedded adapter during offline build
// on release-baseline toolchains (glibc <= 2.35), while allowing newer generic
// Linux hosts whose distro libstdc++.a links newer glibc symbols to build cleanly.
fn check_glibc_floor(path: &std::path::Path) {
    let strict = env::var("ORINFER_STRICT_GLIBC_FLOOR").as_deref() == Ok("1");
    let skip = env::var("ORINFER_SKIP_GLIBC_FLOOR").as_deref() == Ok("1");
    let host_glibc = host_glibc_version();
    let enforce = strict || (!skip && host_glibc.is_none_or(|version| version <= (2, 35)));
    let output = std::process::Command::new("readelf")
        .args(["--version-info", "--wide"])
        .arg(path)
        .output()
        .expect("readelf for the embedded CANN adapter ABI check");
    assert!(
        output.status.success(),
        "Cannot inspect CANN adapter ELF requirements"
    );
    let text = String::from_utf8(output.stdout).expect("readelf UTF-8 output");
    for token in text.split_ascii_whitespace() {
        let token = token.trim_matches(|c: char| c == '(' || c == ')');
        if let Some(version) = token.strip_prefix("GLIBC_") {
            let mut parts = version.split('.').map(str::parse::<u32>);
            if let (Some(Ok(major)), Some(Ok(minor))) = (parts.next(), parts.next())
                && (major, minor) > (2, 34)
            {
                if enforce {
                    panic!(
                        "Embedded CANN adapter requires GLIBC_{version}; release baseline is glibc 2.34"
                    );
                }
                println!(
                    "cargo:warning=Embedded CANN adapter requires GLIBC_{version} (release baseline is glibc 2.34; set ORINFER_STRICT_GLIBC_FLOOR=1 on a glibc <= 2.35 toolchain for portable Ascend release builds)"
                );
                break;
            }
        }
    }
}
