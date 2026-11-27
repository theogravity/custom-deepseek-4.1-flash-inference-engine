// The host oracle uses an exactly zero query head and sink, for which the
// output is the arithmetic mean of visible values plus one zero-valued sink.
#[test]
#[ignore = "requires DS4_TEST_DEVICE on an RTX PRO 6000"]
fn cuda_flash_mla_matches_cpu_mean_and_scalar_attention() -> Result<()> {
    let dev: i32 = std::env::var("DS4_TEST_DEVICE")?.parse()?;
    let sample = |i: usize| {
        let z = (i as u32).wrapping_mul(0x9e3779b9);
        let z = (z ^ (z >> 15)).wrapping_mul(0x735a2d97);
        b16(((z >> 16) as f32 - 32768.) / 24491.)
    };
    for (pos, rows, width) in [
        (0usize, 5usize, 0usize),
        (0, 7, 13),
        (313, 9, 512),
        (371, 257, 65),
    ] {
        // Make the window's first 448 components exactly representable by the
        // serving cache (FP8 QDQ). Rotary components keep their full BF16.
        let window = |rows: usize, seed| {
            let mut v: Vec<_> = (0..rows * DIM).map(|j| sample(j + seed)).collect();
            for row in v.chunks_exact_mut(DIM) {
                fp8_qdq(&mut row[..NOPE], 64);
            }
            v
        };
        let prior = window(2 * WINDOW, 3);
        let current = window(rows, 11);
        let compressed = window(width + 5, 341);
        let mut source = Rows::default();
        pack_values(&compressed, source.packed.for_append());
        let cache = source.on(dev, None)?;
        let to_device_window = |w: &[f32]| {
            upload(
                dev,
                &w.iter()
                    .map(|&x| bf16::from_f32(x).to_bits())
                    .collect::<Vec<_>>(),
            )
        };
        let prior_gpu = to_device_window(&prior)?;
        let current_gpu = to_device_window(&current)?;
        let queries: Vec<_> = (0..rows * HEADS * DIM)
            .map(|j| {
                if (j / DIM) % HEADS == 0 {
                    0.
                } else {
                    sample(j + 97)
                }
            })
            .collect();
        let sinks: Vec<_> = (0..HEADS)
            .map(|h| if h == 0 { 0. } else { sample(h + 119) * 4. })
            .collect();
        let ids: Vec<_> = (0..rows * width)
            .map(|i| {
                let (t, j) = (i / width, i % width);
                if t % 7 == 0 || (t + j) % 13 == 0 {
                    -1
                } else {
                    ((j * 57 + t) % (width + 5)) as i32
                }
            })
            .collect();
        let q = upload(dev, &queries)?;
        let sink = upload(dev, &sinks)?;
        let ids_gpu = upload(dev, &ids)?;
        let out = upload(dev, &vec![f32::NAN; queries.len()])?;
        let scratch = GpuBuffer::new(dev)?;
        // Replay with a fresh window uses pos=0 and a populated compressed
        // source (the second case). Other cases cross ring and query-tile edges.
        let needed = unsafe {
            ds4_attn_flash_bytes(pos as i32, rows as i32, (width + 5) as i32, width as i32)
        };
        scratch.write(&vec![0xffu8; needed])?;
        cuda_ok(
            unsafe {
                ds4_attn_flash(
                    dev,
                    scratch.ptr.as_ptr(),
                    q.data(),
                    prior_gpu.data(),
                    current_gpu.data(),
                    pos as i32,
                    rows as i32,
                    cache,
                    (width + 5) as i32,
                    ids_gpu.data(),
                    width as i32,
                    sink.data(),
                    out.data(),
                )
            },
            "FlashInfer MLA fixture",
        )?;
        assert_eq!(unsafe { ds4_attn_capacity(scratch.ptr.as_ptr()) }, needed);
        let mut actual = vec![0f32; queries.len()];
        read(&out, &mut actual)?;
        // Call the unmodified FlashInfer kernel with an independent host-built
        // page table. It contains the complete ring followed by the chunk:
        // physical IDs and dummy placement differ from the native compact view.
        let host_pages = |values: &[f32]| {
            let mut records = Vec::new();
            pack_values(values, &mut records);
            let n = records.len() / KV_BYTES;
            let mut pages = vec![0u8; (n + 1).div_ceil(64) * 64 * KV_BYTES];
            for (i, row) in records.as_chunks::<KV_BYTES>().0.iter().enumerate() {
                let page = i / 64 * 64 * KV_BYTES;
                let payload = page + (i % 64) * 576;
                let footer = page + 64 * 576 + (i % 64) * 8;
                pages[payload..payload + 576].copy_from_slice(&row[..576]);
                pages[footer..footer + 8].copy_from_slice(&row[576..]);
            }
            upload(dev, &pages)
        };
        let swa = host_pages(&[prior.as_slice(), current.as_slice()].concat())?;
        let extra = host_pages(&compressed)?;
        let padded = width.next_multiple_of(64);
        let mut si = vec![(2 * WINDOW + rows) as i32; rows * WINDOW];
        let mut ei = vec![(width + 5) as i32; rows * padded];
        let (mut sl, mut el) = (Vec::new(), Vec::new());
        for t in 0..rows {
            let end = pos + t + 1;
            sl.push(end.min(WINDOW) as i32);
            for (j, i) in (end.saturating_sub(WINDOW)..end).enumerate() {
                si[t * WINDOW + j] = if i < pos {
                    i % (2 * WINDOW)
                } else {
                    2 * WINDOW + i - pos
                } as i32;
            }
            let chosen: Vec<_> = ids[t * width..(t + 1) * width]
                .iter()
                .copied()
                .filter(|k| *k >= 0)
                .collect();
            ei[t * padded..t * padded + chosen.len()].copy_from_slice(&chosen);
            el.push(chosen.len() as i32);
        }
        let (si, ei, sl, el) = (
            upload(dev, &si)?,
            upload(dev, &ei)?,
            upload(dev, &sl)?,
            upload(dev, &el)?,
        );
        let qb = to_device_window(&queries)?;
        let output = upload(dev, &vec![0xffffu16; queries.len()])?;
        let lse = upload(dev, &vec![f32::NAN; rows * HEADS])?;
        unsafe extern "C" {
            fn ds4_flash_launch(
                q: *const u16,
                swa: *const u8,
                si: *const i32,
                sl: *const i32,
                extra: *const u8,
                ei: *const i32,
                el: *const i32,
                width: i32,
                rows: i32,
                sink: *const f32,
                out: *mut u16,
                lse: *mut f32,
            ) -> i32;
        }
        cuda_ok(
            unsafe {
                ds4_flash_launch(
                    qb.data(),
                    swa.data(),
                    si.data(),
                    sl.data(),
                    extra.data(),
                    ei.data(),
                    el.data(),
                    padded as i32,
                    rows as i32,
                    sink.data(),
                    output.data(),
                    lse.data(),
                )
            },
            "independent FlashInfer page reference",
        )?;
        let mut reference_bits = vec![0u16; queries.len()];
        read(&output, &mut reference_bits)?;
        for (i, (&a, &b)) in actual.iter().zip(&reference_bits).enumerate() {
            assert_eq!(
                a,
                bf16::from_bits(b).to_f32(),
                "FlashInfer page reference pos={pos} row component={i}"
            );
        }
        cuda_ok(
            unsafe {
                ds4_attn_sparse(
                    dev,
                    scratch.ptr.as_ptr(),
                    q.data(),
                    prior_gpu.data(),
                    current_gpu.data(),
                    pos as i32,
                    0,
                    WINDOW as i32,
                    0,
                    cache,
                    ids_gpu.data(),
                    width as i32,
                    sink.data(),
                    rows as i32,
                    out.data(),
                )
            },
            "scalar MLA reference",
        )?;
        let mut reference = vec![0f32; queries.len()];
        read(&out, &mut reference)?;
        let (mut err, mut norm, mut max) = (0f64, 0f64, 0f32);
        for (&a, &b) in actual.iter().zip(&reference) {
            assert!(a.is_finite() && b.is_finite());
            err += f64::from(a - b).powi(2);
            norm += f64::from(b).powi(2);
            max = max.max((a - b).abs());
        }
        let relative = (err / norm).sqrt();
        eprintln!("FlashInfer parity pos={pos} rows={rows} width={width} relative={relative:.6} max={max:.6}");
        assert!(
            // Nonrotary value MMA uses E4M3 probabilities, compared to BF16 in the scalar kernel.
            relative < 0.05 && max < 0.1,
            "relative={relative} max={max}"
        );
        for t in 0..rows {
            let end = pos + t + 1;
            let mut mean = [0f32; DIM];
            let mut count = 1;
            let mut add = |row: &[f32]| {
                for (a, b) in mean.iter_mut().zip(row) {
                    *a += b;
                }
                count += 1;
            };
            for i in end.saturating_sub(WINDOW)..end {
                let row = if i < pos {
                    &prior[(i % (2 * WINDOW)) * DIM..]
                } else {
                    &current[(i - pos) * DIM..]
                };
                add(&row[..DIM]);
            }
            for &id in &ids[t * width..(t + 1) * width] {
                if id >= 0 {
                    add(&compressed[id as usize * DIM..][..DIM]);
                }
            }
            for (j, a) in mean.into_iter().enumerate() {
                let b = actual[t * HEADS * DIM + j];
                assert!(
                    (a / count as f32 - b).abs() < 0.004,
                    "CPU mean at pos={pos} row={t} component={j}: {a}/{count} vs {b}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn flash_prefill_workspace_bounds_query_tiles_and_repacked_history() {
    let bytes = |pos, rows, keys, width| unsafe { ds4_attn_flash_bytes(pos, rows, keys, width) };
    // Beyond the query tile, each additional 64 rows requires one SWA page.
    assert_eq!(
        bytes(9000, 8192, 524288, 512) - bytes(9000, 4096, 524288, 512),
        4096 * KV_BYTES
    );
    assert_eq!(
        bytes(9000, 8192, 524288, 512) - bytes(9000, 8192, 262144, 512),
        262144 * KV_BYTES
    );
    assert_eq!(bytes(128, 256, 0, 0), bytes(524288, 256, 524288, 0));
}
