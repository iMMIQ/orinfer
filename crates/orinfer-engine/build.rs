use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=../../tools/operators/ascend/hccl_graph_registry.cc");
    println!("cargo:rerun-if-changed=../../tools/operators/ascend/hccl_graph_registry_abi.h");
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
                "-static-libgcc",
                "-Wl,--exclude-libs,ALL",
                "-Wl,-z,defs",
                "-Wall",
                "-Wextra",
                "-Werror",
                "../../tools/operators/ascend/hccl_graph_registry.cc",
                "-ldl",
                "-lpthread",
                "-o",
            ])
            .arg(&output)
            .status()
            .expect("C++ compiler for the CANN 9.2 Graph registry adapter");
        assert!(status.success(), "CANN Graph registry adapter build failed");
    } else {
        std::fs::write(&output, []).unwrap();
    }
    println!("cargo:rustc-env=ORINFER_HCCL_REGISTRY={}", output.display());
}
