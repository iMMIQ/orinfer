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

fn snapshot(model: &ModelRuntime, position: usize) -> BTreeMap<String, Vec<u8>> {
    model
        .manifest
        .reset_buffers
        .iter()
        .filter(|name| name.starts_with('L'))
        .map(|name| {
            let bytes = if name.ends_with("_KPages") || name.ends_with("_VPages") {
                position * 4 * 256 * 2
            } else {
                model.execution.sizes[name]
            };
            (name.clone(), download(model, name, bytes))
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
                        let expected: &BTreeMap<String, Vec<u8>> =
                            reference_state.as_ref().unwrap();
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
