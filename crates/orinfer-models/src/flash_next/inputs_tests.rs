#[test]
fn execution_purpose_separates_overlapping_input_shapes() {
    let prefill = [crate::model::PrefillPlan {
        chunk_tokens: 4096,
        min_tokens: 1,
        prefill_program: "prefill".into(),
        head_program: "head".into(),
        stages: vec!["prefill/stage0".into()],
        live_buffers: vec![],
    }];
    let verification: Vec<_> = (2..=8)
        .map(|tokens| orinfer_model_sdk::mtp::VerificationPlan {
            tokens,
            min_tokens: 0,
            program: format!("verify_m{tokens}"),
            restore_program: format!("restore_m{tokens}"),
            capture_program: format!("capture_m{tokens}"),
        })
        .collect();
    for plan in &verification {
        assert_eq!(
            layout(&prefill, &verification, "prefill", plan.tokens).unwrap(),
            ("target", 4096)
        );
        assert_eq!(
            layout(&prefill, &verification, &plan.program, plan.tokens).unwrap(),
            ("target", plan.tokens)
        );
        assert!(layout(&prefill, &verification, &plan.program, plan.tokens + 1).is_err());
    }
    assert_eq!(
        layout(&prefill, &verification, "batch_target", 128).unwrap(),
        ("target", 128)
    );
    assert_eq!(
        layout(&prefill, &verification, "prefill", 421).unwrap(),
        ("target", 4096)
    );
    assert_eq!(
        layout(&prefill, &verification, "target", 1).unwrap(),
        ("target", 1)
    );
    for program in ["prefill", "prefill/stage0"] {
        assert_eq!(
            layout(&prefill, &verification, program, 1).unwrap(),
            ("target", 4096)
        );
        assert!(layout(&prefill, &verification, program, 0).is_err());
        assert!(layout(&prefill, &verification, program, 4097).is_err());
    }
    assert_eq!(
        layout(&prefill, &verification, "mtp_warm_m4", 4).unwrap(),
        ("mtp_warm_m4", 4)
    );
}

use super::*;
use safetensors::tensor::TensorView;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Rows(PathBuf);
impl Rows {
    fn new(encoded: bool) -> (Self, Part, usize) {
        let path = std::env::temp_dir().join(format!(
            "orin-cpu-rows-{}-{}.safetensors",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let width = if encoded { 160 } else { 2560 };
        let columns = if encoded { 2 + width / 4 } else { width };
        let data = vec![0u8; 2 * columns];
        let scales = [0u8; 4];
        let dtype = if encoded { Dtype::U8 } else { Dtype::I8 };
        let mut tensors = vec![(
            "rows",
            TensorView::new(dtype, vec![2, columns], &data).unwrap(),
        )];
        if !encoded {
            tensors.push((
                "scale",
                TensorView::new(Dtype::F16, vec![2], &scales).unwrap(),
            ));
        }
        let bytes = safetensors::serialize(tensors, None).unwrap();
        let (header, metadata) = SafeTensors::read_metadata(&bytes).unwrap();
        let offset = |name| (8 + header + metadata.info(name).unwrap().data_offsets.0) as u64;
        let part = Part {
            file: path.file_name().unwrap().to_str().unwrap().into(),
            sha256: hex::encode(Sha256::digest(&bytes)),
            offset: offset("rows"),
            first: 0,
            rows: 2,
            book: encoded.then_some(0),
            scale_offset: (!encoded).then(|| offset("scale")),
        };
        std::fs::write(&path, bytes).unwrap();
        (Self(path), part, width)
    }
    fn validate(&self, part: &Part, width: usize, verify: bool) -> Result<()> {
        validate_rows(&mut File::open(&self.0).unwrap(), part, width, verify)
    }
}
impl Drop for Rows {
    fn drop(&mut self) {
        std::fs::remove_file(&self.0).unwrap();
    }
}

#[test]
fn cached_rows_share_immutable_storage_and_survive_eviction() {
    let mut cache = Cache::default();
    let row = Arc::new(vec![0x3c00; 64]);
    let budget = row.len() * 2 + 512;
    cache.put((0, 0), Arc::clone(&row), budget);
    let hit = cache.get((0, 0)).unwrap();
    assert!(Arc::ptr_eq(&hit, &row));
    cache.put((0, 1), Arc::new(vec![0x4000; 64]), budget);
    assert!(cache.get((0, 0)).is_none());
    assert_eq!(hit.as_slice(), &[0x3c00; 64]);
    assert_eq!(cache.get((0, 1)).unwrap().as_slice(), &[0x4000; 64]);
    assert!(cache.bytes <= budget);
}

#[test]
fn fast_e8p_decode_matches_reference_bit_for_bit() {
    let reference_decode = |packet: &[u8], table: &Table| -> Vec<u16> {
        let scale = f16::from_bits(u16::from_le_bytes([packet[0], packet[1]])).to_f32();
        let width = table.signs.len();
        let columns = width / 20;
        let mut rotated = vec![0f32; width];
        for (i, code) in packet[2..].as_chunks::<2>().0.iter().enumerate() {
            let values = &table.values[u16::from_le_bytes([code[0], code[1]]) as usize];
            for j in 0..8 {
                rotated[i * 8 + j] = values[j] as f32 * scale;
            }
        }
        let mut out = vec![0f32; width];
        let norm = (20f32).sqrt().recip();
        for i in 0..20 {
            for j in 0..20 {
                let sign = if j == 0 {
                    1.
                } else if i == 0 {
                    -1.
                } else {
                    let delta = (j + 19 - i) % 19;
                    if delta == 0 || (1..19).any(|k| k * k % 19 == delta) {
                        1.
                    } else {
                        -1.
                    }
                };
                let coefficient = sign * norm;
                for col in 0..columns {
                    out[i * columns + col] += coefficient * rotated[j * columns + col];
                }
            }
        }
        for row in out.chunks_exact_mut(columns) {
            let mut half = 1;
            while half < columns {
                for block in row.chunks_exact_mut(half * 2) {
                    for j in 0..half {
                        let a = block[j];
                        let b = block[j + half];
                        block[j] = a + b;
                        block[j + half] = a - b;
                    }
                }
                half *= 2;
            }
        }
        let norm = (columns as f32).sqrt().recip();
        out.iter()
            .zip(&table.signs)
            .map(|(&x, &s)| f16::from_f32((x * norm) * s as f32).to_bits())
            .collect()
    };
    let values: Vec<[i8; 8]> = (0..65536u32)
        .map(|code| {
            std::array::from_fn(|j| {
                (((code.wrapping_mul(17).wrapping_add(j as u32 * 31)) % 127) as i8) - 63
            })
        })
        .collect();
    for width in [160usize, 2560usize] {
        let table = Table {
            values: values.clone(),
            signs: (0..width)
                .map(|i| if (i * 7 + 3) % 3 == 0 { -1 } else { 1 })
                .collect(),
        };
        let mut packet = vec![0u8; 2 + width / 4];
        let scale_bits = f16::from_f32(0.03125).to_bits().to_le_bytes();
        packet[0] = scale_bits[0];
        packet[1] = scale_bits[1];
        for (i, chunk) in packet[2..].as_chunks_mut::<2>().0.iter_mut().enumerate() {
            let code = ((i as u32).wrapping_mul(4099).wrapping_add(13)) as u16;
            chunk.copy_from_slice(&code.to_le_bytes());
        }
        let actual = decode(&packet, &table).unwrap();
        let expected = reference_decode(&packet, &table);
        assert_eq!(actual, expected, "width={width} mismatch");
    }
}

#[test]
fn cpu_rows_verify_content_only_when_requested() {
    for encoded in [false, true] {
        let (file, part, width) = Rows::new(encoded);
        file.validate(&part, width, false).unwrap();
        file.validate(&part, width, true).unwrap();
        let mut bytes = std::fs::read(&file.0).unwrap();
        bytes[part.offset as usize] ^= 1;
        std::fs::write(&file.0, bytes).unwrap();
        file.validate(&part, width, false).unwrap();
        assert!(file.validate(&part, width, true).is_err());
    }
}

#[test]
fn cpu_rows_reject_invalid_structure_in_both_modes() {
    for encoded in [false, true] {
        let (file, mut part, width) = Rows::new(encoded);
        for verify in [false, true] {
            assert!(file.validate(&part, width + 8, verify).is_err());
            part.rows = 1;
            assert!(file.validate(&part, width, verify).is_err());
            part.rows = 2;
            part.offset += 1;
            assert!(file.validate(&part, width, verify).is_err());
            part.offset -= 1;
            if let Some(offset) = part.scale_offset {
                part.scale_offset = Some(offset + 1);
                assert!(file.validate(&part, width, verify).is_err());
                part.scale_offset = Some(offset);
            }
            part.book = if encoded { None } else { Some(0) };
            assert!(file.validate(&part, width, verify).is_err());
            part.book = encoded.then_some(0);
        }
        let mut bytes = std::fs::read(&file.0).unwrap();
        bytes.pop();
        std::fs::write(&file.0, bytes).unwrap();
        for verify in [false, true] {
            assert!(file.validate(&part, width, verify).is_err());
        }
    }
}

#[test]
fn lock_unpoisoned_recovers_after_thread_panic() {
    let mu = Arc::new(Mutex::new(41usize));
    let poisoned = Arc::clone(&mu);
    let _ = std::thread::spawn(move || {
        let mut guard = poisoned.lock().unwrap();
        *guard += 1;
        panic!("intentional poison");
    })
    .join();
    assert!(mu.lock().is_err());
    assert_eq!(*lock_unpoisoned(&mu), 42);
}

#[test]
fn row_pool_recovers_after_worker_panic_without_deadlock() {
    let data = Arc::new(LookupData {
        assets: Assets {
            version: 1,
            vocab: 1,
            eos: 0,
            multipliers: vec![1, 1, 1],
            sizes: vec![1; 16],
            offsets: vec![0; 16],
            books: vec![],
            embedding: vec![],
            ple: vec![],
        },
        tables: vec![],
        embedding: vec![],
        ple: vec![],
    });
    let pool = RowPool::new(data);
    // Trigger parallel path (>= PARALLEL_DECODE_THRESHOLD) with invalid shard indices
    // so both workers and caller panic on `sources[shard]`; catch_unwind must convert
    // this into an Err without deadlocking, and a second call must also finish cleanly.
    let bad_tasks: Vec<DecodeTask> = (0..PARALLEL_DECODE_THRESHOLD)
        .map(|i| (false, i, 0))
        .collect();
    let err = pool.decode(bad_tasks.clone()).unwrap_err();
    assert!(err.contains("panicked"), "unexpected error: {err}");
    let err2 = pool.decode(bad_tasks).unwrap_err();
    assert!(err2.contains("panicked"), "unexpected second error: {err2}");
}

#[test]
#[ignore = "Requires prepared Flash CPU row assets and frozen Python input rows"]
fn reference_rows_match_with_eos_and_request_history() {
    let root = std::env::var("ORINFER_FLASH_MODEL").unwrap();
    let golden = std::env::var("ORINFER_FLASH_GOLDEN").unwrap();
    let raw = std::fs::read(Path::new(&root).join("cache/cpu/inputs.json")).unwrap();
    let inputs = Inputs::open(
        Path::new(&root),
        &crate::artifact::FileIdentity {
            file: "cache/cpu/inputs.json".into(),
            tensor: None,
            sha256: hex::encode(Sha256::digest(&raw)),
        },
        true,
    )
    .unwrap();
    let golden: serde_json::Value =
        serde_json::from_slice(&std::fs::read(golden).unwrap()).unwrap();
    let tokens: Vec<u32> = serde_json::from_value(golden["tokens"].clone()).unwrap();
    let mut history: Vec<u32> = serde_json::from_value(golden["history"].clone()).unwrap();
    let expected: Vec<Vec<u16>> = ["embedding", "ple"]
        .into_iter()
        .map(|k| serde_json::from_value(golden[k].clone()).unwrap())
        .collect();
    for (index, &token) in tokens.iter().enumerate() {
        let synchronous = inputs.prepare(&[token], &history).unwrap();
        inputs.prefetch(&[token], &history, 1);
        let rows = inputs.prepare(&[token], &history).unwrap();
        assert_eq!(synchronous, rows);
        for (kind, (_, raw)) in rows.iter().enumerate() {
            let actual: Vec<u16> = raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|p| u16::from_le_bytes([p[0], p[1]]))
                .collect();
            let expected = &expected[kind][index * 2560..(index + 1) * 2560];
            let max = actual
                .iter()
                .zip(expected)
                .map(|(&a, &b)| (f16::from_bits(a).to_f32() - f16::from_bits(b).to_f32()).abs())
                .fold(0f32, f32::max);
            assert!(max <= 0.0005, "input {index}/{kind}: max FP16 error {max}");
        }
        history.push(token);
    }
    assert!(inputs.lookup.embedding_cache.lock().unwrap().bytes <= EMBEDDING_CACHE_BUDGET);
    assert!(inputs.lookup.ple_cache.lock().unwrap().bytes <= PLE_CACHE_BUDGET);
    assert!(inputs.prepare(&[248320], &[]).is_err());
}
