use std::{env, fs, path::PathBuf, process::Command};
fn main() {
    let cuda = PathBuf::from(env::var_os("CUDA_HOME").unwrap_or("/usr/local/cuda".into()));
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-changed=native");
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
                    "-arch=sm_120",
                    "-Xcompiler",
                    "-fPIC",
                    "-c"
                ])
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
