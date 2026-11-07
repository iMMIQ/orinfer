use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=native/ascend/hccl_graph_registry.cc");
    println!("cargo:rerun-if-changed=native/ascend/hccl_graph_registry_abi.h");
    let output = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("liborinfer_hccl_registry.so");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        let compiler = cc::Build::new().cpp(true).get_compiler();
        let status = compiler
            .to_command()
            .args([
                "-shared",
                "-fPIC",
                "-O2",
                "-std=c++17",
                "-D_GLIBCXX_USE_CXX11_ABI=0",
                "-fvisibility=hidden",
                "-static-libstdc++",
                "-shared-libgcc",
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

// A nested shared object has its own version requirements, independent of the
// Rust executable. Reject an incompatible embedded adapter during offline build.
fn check_glibc_floor(path: &std::path::Path) {
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
            if let (Some(Ok(major)), Some(Ok(minor))) = (parts.next(), parts.next()) {
                assert!(
                    (major, minor) <= (2, 34),
                    "Embedded CANN adapter requires GLIBC_{version}; release baseline is glibc 2.34"
                );
            }
        }
    }
}
