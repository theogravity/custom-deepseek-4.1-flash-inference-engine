use crate::{moe::MoeCache, weights::Weights};
use anyhow::{ensure, Context, Result};
use std::{
    collections::HashMap,
    ffi::{c_char, c_int, c_void, CStr},
    path::Path,
    ptr,
};
extern "C" {
    fn ds_upload(
        dev: c_int,
        out: *mut *mut c_void,
        src: *const c_void,
        scales: *const c_void,
        rows: c_int,
        cols: c_int,
        mode: c_int,
    ) -> c_int;
    fn ds_gemm(
        ctx: *mut c_void,
        w: *const c_void,
        x: *const f32,
        y: *mut f32,
        rows: c_int,
        cols: c_int,
        batch: c_int,
        groups: c_int,
        mode: c_int,
    ) -> c_int;
    fn ds_dense_create(device: c_int, out: *mut *mut c_void) -> c_int;
    fn ds_dense_destroy(ctx: *mut c_void);
    fn ds_free(device: c_int, ptr: *mut c_void);
    fn ds_dense_error() -> *const c_char;
    fn ds_set_device(device: c_int) -> c_int;
}
fn checked(code: c_int) -> Result<()> {
    ensure!(
        code == 0,
        "{}",
        unsafe { CStr::from_ptr(ds_dense_error()) }.to_string_lossy()
    );
    Ok(())
}
struct Matrix {
    device: i32,
    ptr: *mut c_void,
    rows: usize,
    cols: usize,
    mode: c_int,
}
impl Drop for Matrix {
    fn drop(&mut self) {
        unsafe { ds_free(self.device, self.ptr) }
    }
}
pub struct Runtime {
    pub weights: Weights,
    pub device: i32,
    pub moe_cache: MoeCache,
    devices: [i32; 2],
    matrices: HashMap<String, Matrix>,
    contexts: HashMap<i32, *mut c_void>,
}
// All CUDA handles and mutable caches belong to the single locked inference worker.
// Native entry points bind the current OS thread to the requested CUDA device.
unsafe impl Send for Runtime {}
impl Runtime {
    pub fn open(path: &Path, devices: [i32; 2]) -> Result<Self> {
        ensure!(
            devices[0] != devices[1],
            "two distinct CUDA devices required"
        );
        let mut rt = Self {
            weights: Weights::open(path)?,
            device: devices[0],
            devices,
            moe_cache: MoeCache::default(),
            matrices: HashMap::new(),
            contexts: HashMap::new(),
        };
        for dev in devices {
            let mut ctx = ptr::null_mut();
            checked(unsafe { ds_dense_create(dev, &mut ctx) })?;
            rt.contexts.insert(dev, ctx);
        }
        Ok(rt)
    }
    pub fn layer(&mut self, layer: usize) -> Result<()> {
        self.device = self.devices[usize::from(layer >= 20)];
        checked(unsafe { ds_set_device(self.device) })
    }
    pub fn linear(&mut self, name: &str, x: &[f32], batch: usize) -> Result<Vec<f32>> {
        self.group_linear(name, x, batch, 1)
    }
    pub fn group_linear(
        &mut self,
        name: &str,
        x: &[f32],
        batch: usize,
        groups: usize,
    ) -> Result<Vec<f32>> {
        if !self.matrices.contains_key(name) {
            let w = self.weights.tensor(name)?;
            ensure!(w.shape.len() == 2, "expected matrix {name}");
            let (rows, cols) = (w.shape[0], w.shape[1]);
            let mut scales = ptr::null();
            let mode = match w.dtype {
                "F32" => 0,
                "BF16" if name == "head.weight" => {
                    ensure!(cols % 32 == 0, "unaligned vocabulary head");
                    3
                }
                "BF16" => 1,
                "F8_E4M3" => {
                    let s = self.weights.tensor(&name.replace(".weight", ".scale"))?;
                    ensure!(
                        s.dtype == "F8_E8M0" && s.shape == [rows.div_ceil(32), cols.div_ceil(32)],
                        "invalid MXFP8 scales {name}"
                    );
                    scales = s.data.as_ptr().cast();
                    2
                }
                other => anyhow::bail!("unsupported matrix {name} {other}"),
            };
            // The packed MXFP8 tensor, scales and expanded BF16 destination coexist during upload.
            let elements = rows.checked_mul(cols).context("dense shape overflow")?;
            let mut transient = elements
                .checked_mul([4, 2, 3, 3][mode as usize])
                .context("dense size overflow")?;
            if mode == 2 {
                transient = transient
                    .checked_add(rows.div_ceil(32) * cols.div_ceil(32))
                    .context("scale size overflow")?;
            }
            if mode == 3 {
                transient = transient
                    .checked_add(elements / 32)
                    .context("head scale size overflow")?;
            }
            self.moe_cache.reserve(self.device, transient)?;
            let mut buf = ptr::null_mut();
            checked(unsafe {
                ds_upload(
                    self.device,
                    &mut buf,
                    w.data.as_ptr().cast(),
                    scales,
                    rows.try_into()?,
                    cols.try_into()?,
                    mode,
                )
            })
            .with_context(|| format!("upload {name}"))?;
            self.matrices.insert(
                name.to_string(),
                Matrix {
                    device: self.device,
                    ptr: buf,
                    rows,
                    cols,
                    mode,
                },
            );
        }
        let w = &self.matrices[name];
        ensure!(
            groups > 0
                && w.rows.is_multiple_of(groups)
                && w.device == self.device
                && batch > 0
                && x.len() == batch * groups * w.cols,
            "input or placement mismatch at {name}"
        );
        ensure!(
            w.mode != 3 || (batch == 1 && groups == 1),
            "vocabulary head expects one row"
        );
        let mut out = vec![0.; batch * w.rows];
        checked(unsafe {
            ds_gemm(
                self.contexts[&self.device],
                w.ptr,
                x.as_ptr(),
                out.as_mut_ptr(),
                w.rows.try_into()?,
                w.cols.try_into()?,
                batch.try_into()?,
                groups.try_into()?,
                w.mode,
            )
        })
        .with_context(|| format!("project {name}"))?;
        let fp32_compressor = [2, 8, 14, 20]
            .iter()
            .any(|layer| name.starts_with(&format!("layers.{layer}.attn.compressor.")))
            && (name.ends_with(".wkv.weight") || name.ends_with(".wgate.weight"));
        if w.mode != 0
            && !fp32_compressor
            && name != "head.weight"
            && !name.ends_with(".ffn.gate.weight")
        {
            for v in &mut out {
                *v = crate::model::bf(*v);
            }
        }
        Ok(out)
    }
}
impl Drop for Runtime {
    fn drop(&mut self) {
        for ctx in self.contexts.values() {
            unsafe { ds_dense_destroy(*ctx) }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires DS4_TEST_DEVICE on an RTX PRO 6000"]
    fn packed_head_uses_independent_bf16_to_e4m3_group_scales() -> Result<()> {
        let device: i32 = std::env::var("DS4_TEST_DEVICE")?.parse()?;
        let mut ctx = ptr::null_mut();
        checked(unsafe { ds_dense_create(device, &mut ctx) })?;
        struct Handle(*mut c_void);
        impl Drop for Handle {
            fn drop(&mut self) {
                unsafe { ds_dense_destroy(self.0) }
            }
        }
        let ctx = Handle(ctx);
        let weights: Vec<_> = (0..3)
            .flat_map(|r| {
                (0..64).map(move |i| {
                    let n = if i < 32 { 1.0625 } else { 1024.0 }; // The first group ties to 1.0 in E4M3.
                    half::bf16::from_f32([n, -n, 0.][r])
                })
            })
            .collect();
        let mut data = ptr::null_mut();
        checked(unsafe {
            ds_upload(
                device,
                &mut data,
                weights.as_ptr().cast(),
                ptr::null(),
                3,
                64,
                3,
            )
        })?;
        let w = Matrix {
            device,
            ptr: data,
            rows: 3,
            cols: 64,
            mode: 3,
        };
        let x: Vec<_> = (0..64)
            .map(|i| if i < 32 { 1. } else { 1. / 64. })
            .collect();
        let mut y = [0.; 3];
        checked(unsafe { ds_gemm(ctx.0, w.ptr, x.as_ptr(), y.as_mut_ptr(), 3, 64, 1, 1, 3) })?;
        assert_eq!(y, [544., -544., 0.]);
        Ok(())
    }
}
