use std::{env, fs, path::PathBuf, process::Command};
fn main() {
    let cuda = PathBuf::from(env::var_os("CUDA_HOME").unwrap_or("/usr/local/cuda".into()));
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-changed=native");
    // The workstation is fixed to Blackwell. The family ISA includes native
    // packed FP8-to-BF16 conversion for the W8A16 tensor-core kernels.
    println!("cargo:rerun-if-env-changed=DS4_CUDA_ARCH");
    let arch = env::var("DS4_CUDA_ARCH").unwrap_or_else(|_| "sm_120f".into());
    assert!(
        matches!(arch.as_str(), "sm_120" | "sm_120f" | "sm_120a"),
        "unsupported Blackwell ISA"
    );
    println!("cargo:rerun-if-env-changed=DS4_RECIPE_DIR");
    let mut recipe_flags = Vec::new();
    if env::var_os("CARGO_FEATURE_RECIPE_PREFILL").is_some() {
        let root = PathBuf::from(
            env::var_os("DS4_RECIPE_DIR").unwrap_or("/mnt/models/dsv41-flash-exl3/recipe".into()),
        );
        let kernel = root.join("kernels/xmoe_do_i8/dof_gemm.cu");
        let includes = root.join("kernels/xmoe");
        println!("cargo:rerun-if-changed={}", kernel.display());
        println!("cargo:rerun-if-changed={}", includes.display());
        println!("cargo:rerun-if-changed=tools/prepare_recipe_kernel.py");
        assert!(
            Command::new("python3")
                .arg("tools/prepare_recipe_kernel.py")
                .arg(&kernel)
                .arg(out.join("diffbot_device.cuh"))
                .status()
                .unwrap()
                .success(),
            "failed to prepare the pinned Diffbot kernel"
        );
        recipe_flags = vec![
            "-DDS4_RECIPE_PREFILL".into(),
            format!("-I{}", out.display()),
            format!("-I{}", includes.display()),
        ];
    }
    let mut objects = Vec::new();
    for entry in fs::read_dir("native").unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|s| s != "cu") {
            continue;
        }
        let object = out.join(path.file_stem().unwrap()).with_extension("o");
        assert!(
            Command::new(cuda.join("bin/nvcc"))
                .args([
                    "-std=c++17",
                    "-O3",
                    &format!("-arch={arch}"),
                    "-Xcompiler",
                    "-fPIC",
                    "-c"
                ])
                .args(
                    if matches!(
                        path.file_name().and_then(|x| x.to_str()),
                        Some("moe.cu" | "recipe_prefill.cu")
                    ) {
                        &recipe_flags[..]
                    } else {
                        &[]
                    }
                )
                .args(
                    if path.ends_with("recipe_prefill.cu") && !recipe_flags.is_empty() {
                        &["--expt-relaxed-constexpr", "--use_fast_math"][..]
                    } else {
                        &[]
                    }
                )
                .arg(&path)
                .arg("-o")
                .arg(&object)
                .status()
                .unwrap()
                .success(),
            "nvcc failed for {}",
            path.display()
        );
        objects.push(object);
    }
    if !objects.is_empty() {
        let lib = out.join("libds4.a");
        let _ = fs::remove_file(&lib);
        assert!(Command::new("ar")
            .arg("crs")
            .arg(lib)
            .args(objects)
            .status()
            .unwrap()
            .success());
        println!("cargo:rustc-link-search=native={}", out.display());
        println!("cargo:rustc-link-lib=static=ds4");
    }
    println!("cargo:rustc-link-search=native={}/lib64", cuda.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}/lib64", cuda.display());
    for lib in ["cudart", "cublas", "stdc++"] {
        println!("cargo:rustc-link-lib={lib}");
    }
}
