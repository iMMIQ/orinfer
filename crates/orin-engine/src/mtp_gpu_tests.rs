//! Real autoregressive MTP acceptance, state recovery and committed-token timing.
use super::*;
use serde::Deserialize;
use serde_json::json;
use std::path::PathBuf;

#[derive(Deserialize)]
struct Fixture {
    model: PathBuf,
    output: PathBuf,
    cases: Vec<Case>,
    repetitions: usize,
    eos: Vec<u32>,
    #[serde(default = "default_graph")]
    cuda_graph: String,
}

#[derive(Deserialize)]
struct Case {
    id: String,
    input_tokens: Vec<u32>,
    max_new_tokens: usize,
    #[serde(default)]
    stop_after: Option<usize>,
    #[serde(default)]
    sampling: Option<crate::sampling::Options>,
    #[serde(default)]
    images: Vec<crate::vision::ImageInput>,
}

fn default_graph() -> String {
    "decode_only".into()
}

#[test]
#[ignore = "Requires real model, prefix fixtures and exclusive GPU lock"]
fn validate_prefix_reuse() {
    let fixture: Fixture = crate::model::read(&PathBuf::from(
        std::env::var("ORIN_PREFIX_FIXTURE").unwrap(),
    ))
    .unwrap();
    assert!(!fixture.output.exists());
    let mut model = ModelRuntime::load_with_options(
        &fixture.model,
        LoadOptions {
            cuda_graph: fixture.cuda_graph.parse().unwrap(),
            ..LoadOptions::default()
        },
    )
    .unwrap();
    let defaults = crate::sampling::Options {
        temperature: 0.,
        ..Default::default()
    };
    let mut rows = vec![];
    for case in &fixture.cases {
        while let Some(entry) = model.prefix_cache.entries.pop_front() {
            model.execution.release_snapshot(entry.snapshot).unwrap();
        }
        model.prefix_cache.bytes = 0;
        let options = case.sampling.as_ref().unwrap_or(&defaults);
        let mut reference = None;
        let mut reference_state = None;
        let mut reference_logits = None;
        for (name, budget) in [
            ("cold", 0),
            ("populate", 12usize << 30),
            ("hit", 12usize << 30),
        ] {
            model.prefix_cache.budget = budget;
            let at = Instant::now();
            let mut tokens = vec![];
            let mut ttft = 0.0;
            model
                .generate(
                    &case.input_tokens,
                    Some(&case.images),
                    case.max_new_tokens,
                    options,
                    || false,
                    |token| {
                        if tokens.is_empty() {
                            ttft = at.elapsed().as_secs_f64();
                        }
                        tokens.push(token);
                        true
                    },
                )
                .unwrap();
            let elapsed = at.elapsed().as_secs_f64();
            let position = case.input_tokens.len() + tokens.len() - 1;
            let state = snapshot(&model, position);
            let logits = model
                .execution
                .download_bytes(
                    &model.manifest.logits,
                    model.execution.sizes[&model.manifest.logits],
                )
                .unwrap();
            if name == "hit" {
                assert_eq!(
                    model.prefix_statistics.cached_tokens,
                    case.input_tokens.len()
                );
                assert_eq!(
                    reference.as_ref().unwrap(),
                    &tokens,
                    "Cached seeded outputs changed: {}",
                    case.id
                );
                assert_eq!(
                    reference_state.as_ref().unwrap(),
                    &state,
                    "Cached sequence state changed: {}",
                    case.id
                );
                assert_eq!(
                    reference_logits.as_ref().unwrap(),
                    &logits,
                    "Cached target logits changed: {}",
                    case.id
                );
            } else {
                if name == "populate" && options.is_greedy() {
                    assert_eq!(
                        reference.as_ref().unwrap(),
                        &tokens,
                        "Cold/populate target changed: {}",
                        case.id
                    );
                    assert_eq!(
                        reference_state.as_ref().unwrap(),
                        &state,
                        "Cold/populate state changed: {}",
                        case.id
                    );
                }
                reference = Some(tokens.clone());
                reference_state = Some(state);
                reference_logits = Some(logits);
            }
            rows.push(json!({"case":case.id,"mode":name,"ttft_s":ttft,"total_s":elapsed,"tokens":tokens,"prefix":model.prefix_statistics}));
            eprintln!("{} {name}: first token {ttft:.4}s", case.id);
        }
        // Cancellation mutates live state, but must leave its immutable
        // cached prompt usable by a subsequent seeded request.
        let emitted = std::cell::Cell::new(0usize);
        let cancelled = model.generate(
            &case.input_tokens,
            Some(&case.images),
            8,
            options,
            || emitted.get() >= 2,
            |_| {
                emitted.set(emitted.get() + 1);
                true
            },
        );
        assert!(cancelled.is_err_and(|e| e.contains("cancelled")));
        let mut replay = vec![];
        model
            .generate(
                &case.input_tokens,
                Some(&case.images),
                case.max_new_tokens,
                options,
                || false,
                |token| {
                    replay.push(token);
                    true
                },
            )
            .unwrap();
        assert_eq!(reference.as_ref().unwrap(), &replay);
        assert_eq!(
            model.prefix_statistics.cached_tokens,
            case.input_tokens.len()
        );
        if !case.images.is_empty() {
            let mut changed = case.images.clone();
            changed[0].pixels[0] += 0.01;
            model
                .generate(
                    &case.input_tokens,
                    Some(&changed),
                    1,
                    options,
                    || false,
                    |_| true,
                )
                .unwrap();
            assert_eq!(
                model.prefix_statistics.cached_tokens, 0,
                "Changed image reused state"
            );
        }
        if case.input_tokens.len() == 8192 && case.images.is_empty() {
            while let Some(entry) = model.prefix_cache.entries.pop_front() {
                model.execution.release_snapshot(entry.snapshot).unwrap();
            }
            model.prefix_cache.bytes = 0;
            let mut extended = case.input_tokens.clone();
            extended.extend_from_slice(&case.input_tokens[..512]);
            model
                .generate(&extended, None, 1, &defaults, || false, |_| true)
                .unwrap();
            assert!(
                model
                    .prefix_cache
                    .entries
                    .iter()
                    .any(|e| e.tokens.len() == 8192)
            );
            let mut branch = case.input_tokens.clone();
            branch.extend_from_slice(&[198; 8]);
            let mut expected = vec![];
            model.prefix_cache.budget = 0;
            model
                .generate(
                    &branch,
                    None,
                    8,
                    &defaults,
                    || false,
                    |t| {
                        expected.push(t);
                        true
                    },
                )
                .unwrap();
            let state = snapshot(&model, branch.len() + 7);
            model.prefix_cache.budget = 12usize << 30;
            let mut actual = vec![];
            model
                .generate(
                    &branch,
                    None,
                    8,
                    &defaults,
                    || false,
                    |t| {
                        actual.push(t);
                        true
                    },
                )
                .unwrap();
            assert_eq!(model.prefix_statistics.cached_tokens, 8192);
            assert_eq!(expected, actual, "Intermediate checkpoint outputs changed");
            assert_eq!(state, snapshot(&model, branch.len() + 7));
            rows.push(
                json!({"case":case.id,"mode":"checkpoint_branch","prefix":model.prefix_statistics}),
            );
            // Remove the just-cached branch so the next test exercises a
            // shorter prefix again, rather than a complete prompt hit.
            let index = model
                .prefix_cache
                .entries
                .iter()
                .position(|e| e.tokens == branch)
                .unwrap();
            let entry = model.prefix_cache.entries.remove(index).unwrap();
            model.prefix_cache.bytes -= entry.bytes;
            model.execution.release_snapshot(entry.snapshot).unwrap();
        }
        if case.images.is_empty() && case.input_tokens.len().is_multiple_of(512) {
            // These aligned suffixes preserve the cold execution partition,
            // isolating state reuse from the engine's mixed-precision dispatch.
            for suffix in [&[198u32; 8][..], &[1103u32; 8][..]] {
                let mut input = case.input_tokens.clone();
                input.extend_from_slice(suffix);
                let mut expected = None;
                let mut expected_state = None;
                for budget in [0, 12usize << 30] {
                    model.prefix_cache.budget = budget;
                    let mut tokens = vec![];
                    model
                        .generate(
                            &input,
                            None,
                            8,
                            &defaults,
                            || false,
                            |token| {
                                tokens.push(token);
                                true
                            },
                        )
                        .unwrap();
                    let state = snapshot(&model, input.len() + tokens.len() - 1);
                    if budget == 0 {
                        expected = Some(tokens);
                        expected_state = Some(state);
                    } else {
                        assert_eq!(
                            model.prefix_statistics.cached_tokens,
                            case.input_tokens.len()
                        );
                        assert_eq!(
                            expected.as_ref().unwrap(),
                            &tokens,
                            "Prefix branch outputs changed"
                        );
                        assert_eq!(
                            expected_state.as_ref().unwrap(),
                            &state,
                            "Prefix branch state changed"
                        );
                        rows.push(json!({"case":case.id,"mode":"append","prefix":model.prefix_statistics}));
                    }
                }
            }
        }
    }
    std::fs::write(
        fixture.output,
        serde_json::to_vec_pretty(&json!({"status":"passed","requests":rows})).unwrap(),
    )
    .unwrap();
}

fn download(model: &ModelRuntime, name: &str, bytes: usize) -> Vec<u8> {
    assert!(bytes <= model.execution.sizes[name]);
    let mut raw = vec![0u8; bytes];
    // SAFETY: The completed graph owns the source; both extents are checked.
    unsafe {
        check(
            (model.execution.session.driver.download)(
                raw.as_mut_ptr().cast(),
                model.execution.pointers[name],
                bytes,
            ),
            "MTP test state download",
        )
        .unwrap();
    }
    raw
}

fn snapshot(model: &ModelRuntime, position: usize) -> BTreeMap<String, String> {
    use sha2::{Digest, Sha256};
    model
        .manifest
        .reset_buffers
        .iter()
        .filter(|name| name.starts_with('L'))
        .map(|name| {
            let stride = model
                .manifest
                .kv_cache
                .as_ref()
                .and_then(|kv| kv.buffers.get(name));
            let bytes = if let Some(stride) = stride {
                position.checked_mul(*stride).unwrap()
            } else if name.ends_with("_KPages") || name.ends_with("_VPages") {
                position * 4 * 256 * 2
            } else {
                model.execution.sizes[name]
            };
            assert!(bytes <= model.execution.sizes[name]);
            // Hash in bounded chunks: a 256k context has GiB of live KV.
            // Keeping several complete host copies would exhaust Orin RAM.
            let mut digest = Sha256::new();
            let mut scratch = vec![0u8; bytes.min(8 * 1024 * 1024)];
            for offset in (0..bytes).step_by(scratch.len()) {
                let count = scratch.len().min(bytes - offset);
                let pointer = model.execution.pointers[name]
                    .checked_add(offset as u64)
                    .unwrap();
                // SAFETY: Both ranges are checked and the graph has completed.
                unsafe {
                    check(
                        (model.execution.session.driver.download)(
                            scratch.as_mut_ptr().cast(),
                            pointer,
                            count,
                        ),
                        "MTP state hash download",
                    )
                    .unwrap();
                }
                digest.update(&scratch[..count]);
            }
            (name.clone(), format!("{:x}", digest.finalize()))
        })
        .collect()
}

#[test]
#[ignore = "Requires real MTP model, tokenized chat fixtures and exclusive GPU lock"]
fn validate_mtp_generation() {
    let path = PathBuf::from(std::env::var("ORIN_MTP_FIXTURE").unwrap());
    let fixture: Fixture = crate::model::read(&path).unwrap();
    assert!(!fixture.cases.is_empty() && fixture.repetitions > 0);
    assert!(!fixture.output.exists());
    let mut model = ModelRuntime::load_with_options(
        &fixture.model,
        LoadOptions {
            cuda_graph: fixture.cuda_graph.parse().unwrap(),
            ..LoadOptions::default()
        },
    )
    .unwrap();
    let spec = model.manifest.mtp.clone().expect("Native MTP plan");
    let defaults = crate::sampling::Options {
        temperature: 0.,
        ..Default::default()
    };
    let mut rows = vec![];
    let mut deterministic = BTreeMap::new();
    // Interrupt prefill/warm and an in-progress generation, then reuse the
    // same resident model. Subsequent state comparisons prove request reset.
    let first = &fixture.cases[0];
    for cancel_at in [2, 20, usize::MAX] {
        let checks = std::cell::Cell::new(0usize);
        let emitted = std::cell::Cell::new(0usize);
        let result = model.generate(
            &first.input_tokens,
            Some(&first.images),
            32,
            &defaults,
            || {
                checks.set(checks.get() + 1);
                checks.get() >= cancel_at || (cancel_at == usize::MAX && emitted.get() >= 2)
            },
            |_| {
                emitted.set(emitted.get() + 1);
                true
            },
        );
        assert!(
            result.as_ref().is_err_and(|e| e.contains("cancelled")),
            "Cancellation was ignored: {result:?}"
        );
    }
    // Alternating modes and fixtures exercises request isolation. Warmup is
    // excluded from performance summaries. Every call generates real tokens.
    for repetition in 0..=fixture.repetitions {
        for case in &fixture.cases {
            let options = case.sampling.as_ref().unwrap_or(&defaults);
            let mut reference = None;
            let mut reference_state = None;
            for use_mtp in if repetition % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                model.manifest.mtp = use_mtp.then(|| spec.clone());
                let start = Instant::now();
                let mut output = vec![];
                let mut times = vec![];
                let count = model
                    .generate(
                        &case.input_tokens,
                        Some(&case.images),
                        case.max_new_tokens,
                        options,
                        || false,
                        |token| {
                            output.push(token);
                            times.push(start.elapsed().as_secs_f64());
                            !fixture.eos.contains(&token)
                                && case.stop_after.is_none_or(|n| output.len() < n)
                        },
                    )
                    .unwrap();
                let full_s = start.elapsed().as_secs_f64();
                let mtp_stats = model.speculation_statistics.clone();
                assert_eq!(count, output.len());
                assert_eq!(
                    model.read_control(&model.manifest.token).unwrap() as u32,
                    *output.last().unwrap()
                );
                let position = case.input_tokens.len() + count - 1;
                assert_eq!(
                    model.read_control(&model.manifest.position).unwrap() as usize,
                    position
                );
                if use_mtp {
                    assert_eq!(
                        model.read_control(&spec.position).unwrap() as usize,
                        position
                    );
                }
                // Download only after the measured generation finishes.
                let state = snapshot(&model, position);
                if options.temperature == 0.0
                    && let Some(expected) = &reference
                {
                    assert!(expected == &output, "Target outputs differ for {}", case.id);
                    for (name, value) in &state {
                        let expected: &BTreeMap<String, String> = reference_state.as_ref().unwrap();
                        assert!(
                            value == &expected[name],
                            "Target state differs: {} {name}",
                            case.id
                        );
                    }
                } else {
                    reference = Some(output.clone());
                    reference_state = Some(state.clone());
                }
                if use_mtp {
                    if let Some(expected) = deterministic.get(&case.id) {
                        assert_eq!(expected, &output, "Seeded MTP differs: {}", case.id);
                    } else {
                        deterministic.insert(case.id.clone(), output.clone());
                    }
                    // Replay the exact committed history through ordinary decode.
                    // Stochastic MTP and plain sampling need not share token IDs;
                    // their states must agree for the same accepted token history.
                    model.manifest.mtp = None;
                    model
                        .generate(
                            &case.input_tokens,
                            Some(&case.images),
                            1,
                            options,
                            || false,
                            |_| true,
                        )
                        .unwrap();
                    for &token in &output[..output.len() - 1] {
                        model.upload_ids(&model.manifest.token, &[token]).unwrap();
                        model
                            .launch_program("decode", ExecutionPhase::Decode)
                            .unwrap();
                    }
                    let replay = snapshot(&model, position);
                    for (name, value) in &state {
                        assert!(
                            value == &replay[name],
                            "Committed-history replay state differs: {} {name}",
                            case.id
                        );
                    }
                    assert_eq!(
                        model.read_control(&model.manifest.position).unwrap() as usize,
                        position
                    );
                    model.manifest.mtp = Some(spec.clone());
                    if !case.images.is_empty() {
                        let vision = model.manifest.vision.as_ref().unwrap();
                        let (index, _) = vision
                            .layout(&case.input_tokens, &case.images, model.manifest.max_context)
                            .unwrap();
                        let shifted = spec.feature_index.as_ref().unwrap();
                        let actual = model.read_controls(shifted, index.len()).unwrap();
                        let expected: Vec<u32> = index
                            .iter()
                            .skip(1)
                            .copied()
                            .chain(std::iter::once(-1))
                            .map(|x| x as u32)
                            .collect();
                        assert_eq!(actual, expected, "Shifted image inputs differ");
                        // Probe the real draft embedding of the first image token.
                        // The warm graph must copy the visual feature, not the
                        // vocabulary embedding of the image placeholder ID.
                        let first = index.iter().position(|&x| x == 0).unwrap();
                        let feature = download(&model, &vision.features, vision.hidden * 2);
                        model
                            .upload_ids(&spec.position, &[(first - 1) as u32])
                            .unwrap();
                        model
                            .mtp_warm(
                                &spec,
                                &[case.input_tokens[first]],
                                &|| false,
                                ExecutionPhase::Decode,
                            )
                            .unwrap();
                        assert!(
                            download(&model, "MtpEmbedding", feature.len()) == feature,
                            "MTP did not use the visual embedding"
                        );
                    }
                }
                let decode_s = times.last().unwrap() - times[0];
                let tps = (count > 1).then(|| (count - 1) as f64 / decode_s);
                let row = json!({"case":case.id,"mtp":use_mtp,"repetition":repetition,
                    "warmup":repetition==0,"input_tokens":case.input_tokens.len(),
                    "output_tokens":output,"token_times_s":times,"ttft_s":times[0],
                    "decode_s":decode_s,"decode_tps":tps,"generation_s":full_s,
                    "statistics":mtp_stats,
                    "sampling":options,"images":case.images.len(),"cuda_graph":fixture.cuda_graph,
                    "target_position":position,"committed_history_state_equal":true,
                    "greedy_outputs_equal":options.temperature == 0.0});
                eprintln!(
                    "{} mtp={use_mtp} repeat={repetition} {count} tokens {tps:?} TPS",
                    case.id
                );
                rows.push(row);
                std::fs::write(&fixture.output, serde_json::to_vec_pretty(&json!({
                    "status":"running","seed":crate::sampling::EVALUATION_SEED,
                    "manifest_sha256":model.stats.manifest_sha256,"rows":rows,
                    "scope":"Real greedy and sampled autoregressive generation; committed output tokens only. Decode excludes first token and uses callback arrival times. Teacher-forced replay validates state outside measured generation."})).unwrap()).unwrap();
            }
        }
    }
    let mut result: serde_json::Value = crate::model::read(&fixture.output).unwrap();
    result["status"] = json!("passed");
    std::fs::write(&fixture.output, serde_json::to_vec_pretty(&result).unwrap()).unwrap();
}
