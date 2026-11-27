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
    let flash = env::var_os("CARGO_FEATURE_FLASHINFER_PREFILL").is_some();
    let mut flash_flags = Vec::new();
    if flash {
        println!("cargo:rerun-if-env-changed=DS4_FLASHINFER_DIR");
        let root = PathBuf::from(env::var_os("DS4_FLASHINFER_DIR").expect(
            "flashinfer-prefill needs DS4_FLASHINFER_DIR pointing to FlashInfer's csrc and include",
        ));
        let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap())
            .join("tools/flashinfer-sm120.sha256");
        println!("cargo:rerun-if-changed={}", manifest.display());
        for line in fs::read_to_string(&manifest).unwrap().lines() {
            let file = line.split_once("  ").unwrap().1;
            println!("cargo:rerun-if-changed={}", root.join(file).display());
        }
        assert!(
            Command::new("sha256sum")
                .args(["--check", "--status"])
                .arg(&manifest)
                .current_dir(&root)
                .status()
                .unwrap()
                .success(),
            "FlashInfer prefill sources differ from 946200de"
        );
        // Retain the unmodified device kernels and raw launch templates. The
        // general dispatcher instantiates unused architectures and head counts.
        let source = fs::read_to_string(root.join("csrc/sparse_mla_sm120_prefill.cu")).unwrap();
        let launchers = source
            .split_once("// swapAB (warp specialized, 64 heads/CTA), DSV3_2 family.")
            .unwrap()
            .0;
        fs::write(
            out.join("flashinfer_launchers.cuh"),
            format!("{launchers}\n}} }}\n"),
        )
        .unwrap();
        flash_flags = vec![
            format!("-I{}", root.join("include").display()),
            format!("-I{}", out.display()),
            "--expt-relaxed-constexpr".into(),
            "--use_fast_math".into(),
            "-DNDEBUG".into(),
        ];
    }
    let mut objects = Vec::new();
    for entry in fs::read_dir("native").unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|s| s != "cu") {
            continue;
        }
        if path.ends_with("flash_mla.cu") && !flash {
            continue;
        }
        // Architecture-specific instructions in the prefill kernel require
        // family ISA code only. The default -arch shorthand also emits plain sm_120.
        let target = if path.ends_with("flash_mla.cu") {
            "-gencode=arch=compute_120f,code=sm_120f".to_owned()
        } else {
            format!("-arch={arch}")
        };
        let object = out.join(path.file_stem().unwrap()).with_extension("o");
        assert!(
            Command::new(cuda.join("bin/nvcc"))
                .args(["-std=c++17", "-O3", &target, "-Xcompiler", "-fPIC", "-c"])
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
                .args(if path.ends_with("flash_mla.cu") {
                    &flash_flags[..]
                } else {
                    &[][..]
                })
                .args(if path.ends_with("attention.cu") && flash {
                    &["-DDS4_FLASHINFER_PREFILL"][..]
                } else {
                    &[][..]
                })
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
    for lib in ["cudart", "cublas", "cublasLt", "stdc++"] {
        println!("cargo:rustc-link-lib={lib}");
    }
}
