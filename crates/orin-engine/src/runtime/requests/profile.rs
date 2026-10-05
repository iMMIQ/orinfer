//! Ignored diagnostic: full real-model decode iterations and hot GPU graphs.
use super::*;
use crate::execution::Invocation;
use serde_json::json;
use std::cell::RefCell;

thread_local! {
    static MARKS: RefCell<Option<Vec<(&'static str, Instant)>>> = const { RefCell::new(None) };
}
pub(super) fn mark(name: &'static str) {
    MARKS.with(|m| {
        if let Some(m) = m.borrow_mut().as_mut() {
            m.push((name, Instant::now()));
        }
    });
}
fn median(values: &[f64]) -> f64 {
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}
fn category(model: &ModelRuntime, op: &Invocation) -> String {
    match &op.operation {
        crate::model::Operation::Kernel { name } => {
            let parts: Vec<_> = name.split('/').collect();
            if let Some(layer) = parts
                .get(1)
                .and_then(|s| s.strip_prefix("layer"))
                .and_then(|s| s.parse::<usize>().ok())
            {
                format!(
                    "{}/{}/{}",
                    model.manifest.batch_layout.as_ref().unwrap().layers[layer],
                    if parts[0].starts_with("batch_gdn_") {
                        "batched_mixer"
                    } else if op.sequence.is_some() {
                        "private"
                    } else {
                        "shared"
                    },
                    parts[2]
                )
            } else {
                name.clone()
            }
        }
        crate::model::Operation::Copy {
            source,
            destination,
            ..
        } => {
            if destination.ends_with("_History") {
                "copy/conv_history".into()
            } else {
                format!("copy/{source}/{destination}")
            }
        }
        crate::model::Operation::Zero { destination, .. } => format!("zero/{destination}"),
    }
}

#[test]
#[ignore = "Requires real prefill model and exclusive GPU experiment lock"]
fn capture_prefill_ffn_inputs() {
    let fixture: serde_json::Value = crate::model::read(&std::path::PathBuf::from(
        std::env::var("ORIN_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    let output = std::path::PathBuf::from(fixture["output"].as_str().unwrap());
    std::fs::create_dir(&output).unwrap();
    let sources: Vec<GenerationInput> = serde_json::from_value(fixture["cases"].clone()).unwrap();
    let mut model = ModelRuntime::load_with_options(
        std::path::Path::new(fixture["model"].as_str().unwrap()),
        LoadOptions {
            cuda_graph: crate::execution::CudaGraphMode::Off,
            prefix_cache_bytes: 0,
        },
    )
    .unwrap();
    model.manifest.mtp = None;
    let mut inputs = BTreeMap::new();
    for source in sources {
        let (chunk, program, _) = model
            .manifest
            .select_prefill_plan(source.input_tokens.len())
            .unwrap();
        let program = program.to_owned();
        model
            .generate(
                &source.input_tokens,
                None,
                1,
                &source.sampling,
                || false,
                |_| false,
            )
            .unwrap();
        model
            .execution
            .reset_sequence(&model.manifest.reset_buffers)
            .unwrap();
        model
            .prepare_visual(&source.input_tokens, &[], &|| false)
            .unwrap();
        // reset_sequence releases demand-mapped KV. Manual kernel stepping
        // needs the same mapping/workspace preparation as submit_program.
        model
            .execution
            .ensure_sequence_program(0, &program)
            .unwrap();
        model
            .upload_ids(&model.manifest.input, &source.input_tokens[..chunk])
            .unwrap();
        for operation in model.manifest.programs[&program].clone() {
            if let crate::model::Operation::Kernel { name } = &operation {
                let spec = model
                    .manifest
                    .kernels
                    .iter()
                    .find(|k| k.name == *name)
                    .unwrap();
                for family in ["GateUp", "Down"] {
                    let prefix = format!("L0_{family}");
                    let expanded = spec.args.iter().any(|a| matches!(a,crate::artifact::Argument::Buffer { name } if name == "TemporaryW8"));
                    if name.starts_with(&format!("{program}/layer0/"))
                        && spec.args.iter().any(|a| matches!(a,crate::artifact::Argument::Buffer { name } if name == &format!("{prefix}_WS")))
                        && spec.args.iter().any(|a| matches!(a,crate::artifact::Argument::Buffer { name } if name == "TemporaryA8"))
                    {
                        let scales = model.manifest.buffers.iter().find(|b| b.name == format!("{prefix}_S")).unwrap();
                        let width = scales.shape[1] * 128;
                        let activation = model.execution.download_bytes("TemporaryA8", chunk * width).unwrap();
                        let scale = model.execution.download_bytes("TemporaryAS", chunk * 2).unwrap();
                        let key = format!("{prefix}_m{chunk}");
                        std::fs::write(output.join(format!("{key}.a8")), &activation).unwrap();
                        std::fs::write(output.join(format!("{key}.scale.f16")), &scale).unwrap();
                        let mut entry = json!({"rows":chunk,"width":width,
                            "activation_sha256":crate::artifact::sha256(&activation),
                            "scale_sha256":crate::artifact::sha256(&scale)});
                        if expanded {
                            let rows = scales.shape[0];
                            let weight = model.execution.download_bytes("TemporaryW8", rows * width).unwrap();
                            std::fs::write(output.join(format!("{key}.w8")), &weight).unwrap();
                            entry["weight_shape"] = json!([rows, width]);
                            entry["weight_sha256"] = json!(crate::artifact::sha256(&weight));
                        }
                        inputs.insert(key, entry);
                    }
                }
            }
            model
                .execution
                .execute_batch(
                    vec![],
                    &[Invocation {
                        operation,
                        sequence: None,
                        launch: None,
                        views: BTreeMap::new(),
                    }],
                    false,
                )
                .unwrap();
        }
    }
    std::fs::write(output.join("capture.json"),serde_json::to_vec_pretty(&json!({
        "fingerprint":model.stats.manifest_sha256,"seed":20261002,"inputs":inputs,
        "scope":"Actual A8 inputs and row scales before first-layer prefill FFN projections; no MTP."
    })).unwrap()).unwrap();
}

#[test]
#[ignore = "Requires real model and exclusive GPU experiment lock"]
fn capture_prefill_attention_inputs() {
    let fixture: serde_json::Value = crate::model::read(&std::path::PathBuf::from(
        std::env::var("ORIN_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    let output = std::path::PathBuf::from(fixture["output"].as_str().unwrap());
    std::fs::create_dir(&output).unwrap();
    let source: GenerationInput = serde_json::from_value(fixture["cases"][0].clone()).unwrap();
    assert!(source.images.is_empty());
    let mut model = ModelRuntime::load_with_options(
        std::path::Path::new(fixture["model"].as_str().unwrap()),
        LoadOptions {
            cuda_graph: crate::execution::CudaGraphMode::Off,
            prefix_cache_bytes: 0,
        },
    )
    .unwrap();
    model.manifest.mtp = None;
    let length = source.input_tokens.len();
    let (chunk, program, _) = model.manifest.select_prefill_plan(length).unwrap();
    let program = program.to_owned();
    let base = length - chunk;
    assert!(base > 0 && base.is_multiple_of(chunk));
    model
        .generate(
            &source.input_tokens[..base],
            None,
            1,
            &source.sampling,
            || false,
            |_| false,
        )
        .unwrap();
    model
        .execution
        .ensure_sequence_program(0, &program)
        .unwrap();
    model
        .upload_ids(&model.manifest.input, &source.input_tokens[base..])
        .unwrap();
    for operation in model.manifest.programs[&program].clone() {
        if let crate::model::Operation::Kernel { name } = &operation {
            let spec = model
                .manifest
                .kernels
                .iter()
                .find(|k| k.name == *name)
                .unwrap();
            let reads = |buffer: &str| {
                spec.args.iter().any(
                    |a| matches!(a,crate::artifact::Argument::Buffer { name } if name == buffer),
                )
            };
            if reads("FullQ") && reads("PrefillK") && reads("PrefillV") {
                let mut records = BTreeMap::new();
                for (name, bytes) in [
                    ("FullQ", chunk * 6144 * 2),
                    ("FullGate", chunk * 6144 * 2),
                    ("PrefillK", length * 1024 * 2),
                    ("PrefillV", length * 1024 * 2),
                    ("Positions", chunk * 4),
                    ("SeqLength", 4),
                ] {
                    let raw = model.execution.download_bytes(name, bytes).unwrap();
                    let file = format!("{name}.bin");
                    std::fs::write(output.join(&file), &raw).unwrap();
                    records.insert(
                        name,
                        json!({"file":file,"bytes":bytes,"sha256":crate::artifact::sha256(&raw)}),
                    );
                }
                std::fs::write(output.join("capture.json"),serde_json::to_vec_pretty(&json!({
                    "fingerprint":model.stats.manifest_sha256,"seed":20261002,
                    "rows":chunk,"context":length,"base":base,"kernel":name,"inputs":records,
                    "scope":"Actual first full-attention layer inputs before the final real-model prefill chunk; no MTP."
                })).unwrap()).unwrap();
                return;
            }
        }
        model
            .execution
            .execute_batch(
                vec![],
                &[Invocation {
                    operation,
                    sequence: None,
                    launch: None,
                    views: BTreeMap::new(),
                }],
                false,
            )
            .unwrap();
    }
    panic!("Prepared model has no staged full-attention prefill reader");
}

#[test]
#[ignore = "Requires real model and exclusive GPU experiment lock"]
fn profile_prefill_stages() {
    let fixture: serde_json::Value = crate::model::read(&std::path::PathBuf::from(
        std::env::var("ORIN_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    let output = std::path::PathBuf::from(fixture["output"].as_str().unwrap());
    assert!(!output.exists());
    let sources: Vec<GenerationInput> = serde_json::from_value(fixture["cases"].clone()).unwrap();
    let mut model = ModelRuntime::load_with_options(
        std::path::Path::new(fixture["model"].as_str().unwrap()),
        LoadOptions {
            cuda_graph: crate::execution::CudaGraphMode::DecodeOnly,
            prefix_cache_bytes: 0,
        },
    )
    .unwrap();
    model.manifest.mtp = None;
    let mut rows = vec![];
    for source in sources {
        let length = source.input_tokens.len();
        let (chunk, program, _) = model.manifest.select_prefill_plan(length).unwrap();
        let program = program.to_owned();
        let base = length - chunk;
        // Materialize the exact context and its KV/workspace before profiling.
        // Replays reset only the target position; they are timing diagnostics,
        // not a comparison of the recurrent state or generated text.
        model
            .generate(
                &source.input_tokens,
                None,
                1,
                &source.sampling,
                || false,
                |_| false,
            )
            .unwrap();
        model
            .upload_ids(&model.manifest.input, &source.input_tokens[base..])
            .unwrap();
        let plan: Vec<_> = model.manifest.programs[&program]
            .iter()
            .cloned()
            .map(|operation| Invocation {
                operation,
                sequence: None,
                launch: None,
                views: BTreeMap::new(),
            })
            .collect();
        let trials = model
            .execution
            .profile_graph_operations_at_position(
                &plan,
                Some((&model.manifest.position, base as u32)),
            )
            .unwrap();
        let mut groups: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        for (index, op) in plan.iter().enumerate() {
            let values = groups
                .entry(category(&model, op))
                .or_insert(vec![0.; trials.len()]);
            for (trial, times) in trials.iter().enumerate() {
                values[trial] += f64::from(times[index]);
            }
        }
        rows.push(json!({"input_tokens":length,"profiled_chunk_tokens":chunk,"context_before_chunk":base,
            "program":program,"operation_count":plan.len(),
            "instrumented_graph_gpu_median_ms":median(&trials.iter().map(|t|t.iter().map(|x|f64::from(*x)).sum()).collect::<Vec<f64>>()),
            "gpu_group_median_ms":groups.iter().map(|(k,v)|(k.clone(),median(v))).collect::<BTreeMap<_,_>>() }));
        std::fs::write(&output,serde_json::to_vec_pretty(&json!({"rows":rows,
            "scope":"Last real-model prefill chunk at fixed target position; three instrumented graph trials after warm-up; head and host submission excluded. Recurrent state advances between replays. Diagnostic only, not end-to-end TPS or quality acceptance."})).unwrap()).unwrap();
    }
}

#[test]
#[ignore = "Captures real FFN inputs; requires model and exclusive GPU experiment lock"]
fn capture_decode_projections() {
    use std::io::Write;
    let fixture: serde_json::Value = crate::model::read(&std::path::PathBuf::from(
        std::env::var("ORIN_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    let output = std::path::PathBuf::from(fixture["output"].as_str().unwrap());
    std::fs::create_dir(&output).unwrap();
    let sources: Vec<GenerationInput> = serde_json::from_value(fixture["cases"].clone()).unwrap();
    let mut model = ModelRuntime::load_with_options(
        std::path::Path::new(fixture["model"].as_str().unwrap()),
        LoadOptions {
            cuda_graph: crate::execution::CudaGraphMode::Off,
            prefix_cache_bytes: 0,
        },
    )
    .unwrap();
    model.manifest.mtp = None;
    let options = scheduler::Options::default();
    let mut samples = 0;
    for source in sources {
        let mut request = model.start_request(source, &|| false).unwrap();
        while request.prefilling {
            model
                .advance_requests(&mut [&mut request], &options)
                .unwrap();
        }
        model.with_request(&mut request, |model, req| {
            model.execution.ensure_sequence_program(req.slot, "decode")?;
            let operations = model.manifest.programs["decode"].clone();
            for _ in 0..8 {
                for operation in &operations {
                    if let crate::model::Operation::Kernel { name } = operation {
                        let pieces: Vec<_> = name.split('/').collect();
                        if let Some(layer) = pieces.get(1).and_then(|s| s.strip_prefix("layer")) {
                            let spec = model.manifest.kernels.iter().find(|k| k.name == *name).unwrap();
                            for (family, input) in [("GateUp", "Norm"), ("Down", "Activated")] {
                                if spec.args.iter().any(|a| matches!(a,
                                    crate::artifact::Argument::Buffer { name } if name == &format!("L{layer}_{family}_P"))) {
                                    if !spec.args.iter().any(|a| matches!(a,
                                        crate::artifact::Argument::Buffer { name } if name == input)) {
                                        return Err("Capture requires a source package with FP16 Norm/Activated inputs".into());
                                    }
                                    let stride = model.manifest.batch_layout.as_ref().unwrap().row_strides[input];
                                    let bytes = model.execution.download_bytes(input, stride)?;
                                    let mut file = std::fs::OpenOptions::new().create(true).append(true)
                                        .open(output.join(format!("L{layer}_{family}.f16"))).map_err(|e| e.to_string())?;
                                    file.write_all(&bytes).map_err(|e| e.to_string())?;
                                }
                            }
                        }
                    }
                    model.execution.execute_batch(vec![], &[Invocation {
                        operation: operation.clone(), sequence: None, launch: None, views: BTreeMap::new(),
                    }], false)?;
                }
                model.execution.sync()?;
                samples += 1;
            }
            Ok(())
        }).unwrap();
        model.finish_request(&mut request, false).unwrap();
    }
    std::fs::write(
        output.join("capture.json"),
        serde_json::to_vec_pretty(&json!({
            "model":fixture["model"],"fingerprint":model.stats.manifest_sha256,
            "samples_per_projection":samples,"seed":20261002,
            "scope":"Real sequential target-model decode inputs; no MTP."
        }))
        .unwrap(),
    )
    .unwrap();
}

#[test]
#[ignore = "Requires real model and exclusive GPU experiment lock"]
fn profile_continuous_decode() {
    let fixture: serde_json::Value = crate::model::read(&std::path::PathBuf::from(
        std::env::var("ORIN_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    let output = std::path::PathBuf::from(fixture["output"].as_str().unwrap());
    assert!(!output.exists());
    let sources: Vec<GenerationInput> = serde_json::from_value(fixture["cases"].clone()).unwrap();
    let mut model = ModelRuntime::load_with_options(
        std::path::Path::new(fixture["model"].as_str().unwrap()),
        LoadOptions {
            cuda_graph: fixture["cuda_graph"].as_str().unwrap().parse().unwrap(),
            prefix_cache_bytes: 0,
        },
    )
    .unwrap();
    // Isolate target decode. This is not a quality assessment or API benchmark.
    model.manifest.mtp = None;
    let options = scheduler::Options::default();
    let mut rows = vec![];
    let batches: Vec<usize> = fixture
        .get("batches")
        .map(|v| serde_json::from_value(v.clone()).unwrap())
        .unwrap_or_else(|| vec![1, 2, 4, 8, 16, 32]);
    assert!(batches.iter().all(|&b| (1..=128).contains(&b)));
    for count in batches {
        let mut requests: Vec<_> = (0..count)
            .map(|i| {
                let source = &sources[i % sources.len()];
                model
                    .start_request(
                        GenerationInput {
                            input_tokens: source.input_tokens.clone(),
                            images: vec![],
                            max_new_tokens: 160,
                            sampling: source.sampling.clone(),
                            prefix_hints: vec![],
                        },
                        &|| false,
                    )
                    .unwrap()
            })
            .collect();
        while requests.iter().any(|r| r.prefilling) {
            model
                .advance_requests(&mut requests.iter_mut().collect::<Vec<_>>(), &options)
                .unwrap();
        }
        for _ in 0..2 {
            model
                .advance_requests(&mut requests.iter_mut().collect::<Vec<_>>(), &options)
                .unwrap();
        }
        let mut walls = vec![];
        let mut phases: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        for _ in 0..64 {
            MARKS.with(|m| *m.borrow_mut() = Some(vec![]));
            let started = Instant::now();
            let steps = model
                .advance_requests(&mut requests.iter_mut().collect::<Vec<_>>(), &options)
                .unwrap();
            walls.push(started.elapsed().as_secs_f64() * 1000.);
            assert_eq!(steps.iter().map(|s| s.tokens.len()).sum::<usize>(), count);
            let marks = MARKS.with(|m| m.borrow_mut().take().unwrap());
            for pair in marks.windows(2) {
                phases
                    .entry(pair[1].0.into())
                    .or_default()
                    .push(pair[1].1.duration_since(pair[0].1).as_secs_f64() * 1000.);
            }
        }
        let segments: Vec<_> = requests
            .iter()
            .map(|r| BatchSegment {
                slot: r.slot,
                tokens: 1,
            })
            .collect();
        let key: Vec<_> = segments.iter().map(|s| (s.slot, s.tokens)).collect();
        // Profiling replays advance state beyond the CPU histories. Reserve the
        // bounded extra steps before replay, including a possible slab boundary.
        for request in &requests {
            model
                .execution
                .ensure_sequence_program(request.slot, "prefill_m32")
                .unwrap();
        }
        let gpu: Vec<f64> = model
            .execution
            .profile_hot_graph(&key, 16)
            .unwrap()
            .into_iter()
            .map(f64::from)
            .collect();
        let plan = if count == 1 {
            model.manifest.programs["decode"]
                .iter()
                .cloned()
                .map(|operation| Invocation {
                    operation,
                    sequence: None,
                    launch: None,
                    views: BTreeMap::new(),
                })
                .collect()
        } else {
            crate::architecture::batch_plan(&model.manifest, &segments).unwrap()
        };
        let trials = model.execution.profile_graph_operations(&plan).unwrap();
        let mut groups: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        for (index, op) in plan.iter().enumerate() {
            let values = groups
                .entry(category(&model, op))
                .or_insert(vec![0.; trials.len()]);
            for (trial, times) in trials.iter().enumerate() {
                values[trial] += f64::from(times[index]);
            }
        }
        let row = json!({"batch":count,"iterations":64,"wall_median_ms":median(&walls),
            "target_tokens_per_second":count as f64*1000./(walls.iter().sum::<f64>()/walls.len() as f64),
            "phases_median_ms":phases.iter().map(|(k,v)|(k.clone(),median(v))).collect::<BTreeMap<_,_>>(),
            "hot_graph_gpu_median_ms":median(&gpu),"operation_count":plan.len(),
            "instrumented_graph_gpu_median_ms":median(&trials.iter().map(|t| t.iter().map(|x|f64::from(*x)).sum()).collect::<Vec<f64>>()),
            "gpu_group_median_ms":groups.iter().map(|(k,v)|(k.clone(),median(v))).collect::<BTreeMap<_,_>>()});
        eprintln!("PROFILE {row}");
        rows.push(row);
        std::fs::write(&output,serde_json::to_vec_pretty(&json!({"scope":"64 real target decode iterations per fixed batch; no MTP, admission/prefill/API excluded. GPU operation timing uses external-event nodes in an instrumented replay, compare its overhead with the uninstrumented hot graph.","rows":rows})).unwrap()).unwrap();
        for request in &mut requests {
            model.finish_request(request, false).unwrap();
        }
    }
}
