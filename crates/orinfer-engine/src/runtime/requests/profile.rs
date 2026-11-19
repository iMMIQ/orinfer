//! Ignored diagnostics: real-model kernel captures, partition state and GPU graphs.
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

fn row_invocations(model: &ModelRuntime, program: &str, rows: usize) -> Result<Vec<Invocation>> {
    let operations = &model.manifest.programs[program];
    if let Some(template) = model
        .manifest
        .row_execution
        .as_ref()
        .filter(|s| s.programs.contains(program))
    {
        template.specialize(
            program,
            operations,
            &model.manifest.dynamic_batch_kernels,
            rows,
        )
    } else {
        Ok(operations
            .iter()
            .cloned()
            .map(|operation| Invocation {
                operation,
                sequence: None,
                launch: None,
                views: BTreeMap::new(),
            })
            .collect())
    }
}

fn profile_native_ncu(model: &mut ModelRuntime, plan: &[Invocation]) {
    // Keep diagnostic instrumentation out of the production launch path.
    // SAFETY: The library stays loaded until all synchronous NVTX calls end.
    unsafe {
        let library = libloading::Library::new("libnvToolsExt.so.1").unwrap();
        let push = library
            .get::<unsafe extern "C" fn(*const std::ffi::c_char) -> i32>(b"nvtxRangePushA\0")
            .unwrap();
        let pop = library
            .get::<unsafe extern "C" fn() -> i32>(b"nvtxRangePop\0")
            .unwrap();
        crate::backend::cuda::check(
            cudarc::driver::sys::cuProfilerStart(),
            "profile decode chain",
        )
        .unwrap();
        for (index, operation) in plan.iter().enumerate() {
            let label = match &operation.operation {
                crate::model::Operation::Kernel { name } => name.clone(),
                _ => "copy".to_owned(),
            };
            let label =
                std::ffi::CString::new(format!("{index:04}|{}", label.replace('/', "|"))).unwrap();
            push(label.as_ptr());
            let launched = model.execution.execute_batch(
                vec![],
                std::slice::from_ref(operation),
                false,
                false,
            );
            pop();
            launched.unwrap();
        }
        model.execution.sync().unwrap();
        crate::backend::cuda::check(
            cudarc::driver::sys::cuProfilerStop(),
            "stop decode chain profile",
        )
        .unwrap();
    }
}

#[test]
#[ignore = "Captures real kernel arguments; requires model and exclusive GPU experiment lock"]
fn capture_kernel_arguments() {
    let fixture: serde_json::Value = crate::model::read(&std::path::PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    let output = std::path::PathBuf::from(fixture["output"].as_str().unwrap());
    std::fs::create_dir(&output).unwrap();
    let targets: BTreeSet<String> = serde_json::from_value(fixture["kernels"].clone()).unwrap();
    assert!(!targets.is_empty());
    let mut model = ModelRuntime::load_with_options(
        std::path::Path::new(fixture["model"].as_str().unwrap()),
        LoadOptions {
            cuda_graph: crate::execution::CudaGraphMode::Off,
            prefix_cache_bytes: 0,
            mtp_drafts: Some(0),
            ..LoadOptions::default()
        },
    )
    .unwrap();
    let mut captures = vec![];
    for phase in ["prefill", "decode"] {
        let source: GenerationInput = serde_json::from_value(fixture["cases"][0].clone()).unwrap();
        assert!(source.images.is_empty());
        let mut request = model.start_request(source, &|| false).unwrap();
        if phase == "decode" {
            while request.prefilling {
                model
                    .advance_requests(&mut [&mut request], &scheduler::Options::default())
                    .unwrap();
            }
        }
        model.with_request(&mut request, |model, req| {
            let (tokens, history, program) = if phase == "prefill" {
                let (chunk, program, _) = model.manifest.select_prefill_plan(req.input.len())?;
                if chunk != req.input.len() {
                    return Err("Kernel capture requires a single full prefill chunk".into());
                }
                (&req.input[..], &req.input[..0], program.to_owned())
            } else {
                let last = req.history.len() - 1;
                (&req.history[last..], &req.history[..last], "decode".to_owned())
            };
            model.execution.ensure_sequence_program(req.slot, &program)?;
            model.upload_ids(if phase == "decode" { &model.manifest.token } else { &model.manifest.input }, tokens)?;
            model.upload_segment_controls(tokens.len())?;
            if phase == "prefill" {
                model.prepare_prefill_inputs(&program, tokens, history)?;
            } else {
                model.prepare_inputs(tokens, history)?;
            }
            let mut remaining = targets.iter().filter(|name| name.starts_with(&format!("{program}/"))).cloned().collect::<BTreeSet<_>>();
            for operation in model.manifest.programs[&program].clone() {
                let selected = match &operation {
                    crate::model::Operation::Kernel { name } if remaining.contains(name) => Some(name.clone()),
                    _ => None,
                };
                let mut records = BTreeMap::new();
                if let Some(name) = &selected {
                    let spec = model.manifest.kernels.iter().find(|k| k.name == *name).unwrap();
                    let directory = output.join(name.replace('/', "-"));
                    std::fs::create_dir(&directory).map_err(|e| e.to_string())?;
                    for argument in &spec.args {
                        let buffer_name = match argument {
                            crate::artifact::Argument::Buffer { name } | crate::artifact::Argument::BufferSlice { name, .. } => name,
                            _ => return Err("Kernel capture requires pointer-only arguments".into()),
                        };
                        let b = model.manifest.buffers.iter().find(|b| b.name == *buffer_name).unwrap();
                        if b.data.is_some() || records.contains_key(buffer_name) { continue; }
                        let bytes = model.execution.session.virtual_buffers.borrow().get(buffer_name)
                            .map_or(model.execution.sizes[buffer_name], |v| v.mapped.min(model.execution.sizes[buffer_name]));
                        let raw = model.execution.download_bytes(buffer_name, bytes)?;
                        let file = format!("{}.before.bin", records.len());
                        std::fs::write(directory.join(&file), &raw).map_err(|e| e.to_string())?;
                        records.insert(buffer_name.clone(), json!({"file":file,"bytes":bytes,"sha256":crate::artifact::sha256(&raw)}));
                    }
                }
                model.execution.execute_batch(vec![], &[Invocation { operation, sequence: None, launch: None, views: BTreeMap::new() }], false, false)?;
                if let Some(name) = selected {
                    let directory = output.join(name.replace('/', "-"));
                    for (index, (buffer, record)) in records.iter_mut().enumerate() {
                        let raw = model.execution.download_bytes(buffer, record["bytes"].as_u64().unwrap() as usize)?;
                        let file = format!("{index}.after.bin");
                        std::fs::write(directory.join(&file), &raw).map_err(|e| e.to_string())?;
                        record["after"] = json!({"file":file,"sha256":crate::artifact::sha256(&raw)});
                    }
                    captures.push(json!({"kernel":name,"directory":directory.file_name().unwrap().to_str().unwrap(),"buffers":records}));
                    remaining.remove(&name);
                    if remaining.is_empty() { break; }
                }
            }
            if !remaining.is_empty() { return Err("Selected capture kernel is absent from its program".into()); }
            Ok(())
        }).unwrap();
        model.finish_request(&mut request, false).unwrap();
    }
    assert_eq!(captures.len(), targets.len());
    std::fs::write(output.join("capture.json"), serde_json::to_vec_pretty(&json!({
        "model":fixture["model"],"fingerprint":model.stats.manifest_sha256,"seed":20261002,"captures":captures,
        "scope":"Real first prefill and decode kernel inputs and outputs, MTP off. State is request-private; individual launches for offline profiling, not performance acceptance. Weight payloads remain in the source model."
    })).unwrap()).unwrap();
}

#[test]
#[ignore = "Requires real model, tokenized requests and exclusive GPU experiment lock"]
fn profile_cached_generation() {
    let fixture: serde_json::Value = crate::model::read(&std::path::PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    let output = std::path::PathBuf::from(fixture["output"].as_str().unwrap());
    assert!(!output.exists());
    let sources: Vec<GenerationInput> = serde_json::from_value(fixture["cases"].clone()).unwrap();
    let repetitions = fixture["repetitions"].as_u64().unwrap_or(2) as usize;
    let eos: Vec<u32> = serde_json::from_value(fixture["eos"].clone()).unwrap();
    let prefix_cache_bytes = fixture["prefix_cache_bytes"]
        .as_u64()
        .map(|bytes| bytes as usize)
        .unwrap_or(12usize << 30);
    let mut model = ModelRuntime::load_with_options(
        std::path::Path::new(fixture["model"].as_str().unwrap()),
        LoadOptions {
            cuda_graph: crate::execution::CudaGraphMode::DecodeOnly,
            prefix_cache_bytes,
            mtp_drafts: Some(fixture["mtp_drafts"].as_u64().unwrap_or(3) as usize),
            ..LoadOptions::default()
        },
    )
    .unwrap();
    if let Some(regions) = fixture.get("parallel_regions") {
        *model.execution.parallel_regions.borrow_mut() =
            serde_json::from_value(regions.clone()).unwrap();
    }
    model
        .execution
        .graph_profile
        .gate_first
        .set(fixture["parallel_gate_first"].as_bool().unwrap_or(false));
    if fixture.get("parallel_priority").is_some() {
        *model.execution.graph_profile.priority.borrow_mut() = fixture["parallel_priority"]
            .as_str()
            .unwrap_or("off")
            .to_owned();
    }
    let mut rows = vec![];
    let mut deterministic = BTreeMap::new();
    for (case, source) in sources.iter().enumerate() {
        for repetition in 0..=repetitions {
            let tracing = fixture["nsys"].as_bool() == Some(true) && case == 0 && repetition == 1;
            if tracing {
                // SAFETY: Current CUDA context; Nsight owns collection for this diagnostic.
                unsafe {
                    crate::backend::cuda::check(
                        cudarc::driver::sys::cuProfilerStart(),
                        "start diagnostic trace",
                    )
                    .unwrap();
                }
            }
            let started = Instant::now();
            let mut request = model
                .start_request(
                    GenerationInput {
                        audios: source.audios.clone(),
                        input_tokens: source.input_tokens.clone(),
                        images: source.images.clone(),
                        max_new_tokens: source.max_new_tokens,
                        sampling: source.sampling.clone(),
                        prefix_hints: source.prefix_hints.clone(),
                    },
                    &|| false,
                )
                .unwrap();
            let mut tokens = vec![];
            let mut arrivals = vec![];
            let mut first = None;
            let mut first_count = 0;
            let mut ended = false;
            while !request.is_finished() && !ended {
                // Equal proposal widths isolate kernel arithmetic from the
                // normal latency-adaptive profile selection during validation.
                if fixture["fixed_speculation"].as_bool() == Some(true) {
                    model.dynamic_speculation.fixed_width = Some(
                        (fixture["mtp_drafts"].as_u64().unwrap() as usize + 1)
                            .min(request.limit - request.generated),
                    );
                }
                let steps = model
                    .advance_requests(&mut [&mut request], &scheduler::Options::default())
                    .unwrap();
                for step in steps {
                    if step.tokens.is_empty() {
                        continue;
                    }
                    let elapsed = started.elapsed().as_secs_f64();
                    if first.is_none() {
                        first = Some(elapsed);
                        first_count = step.tokens.len();
                    }
                    ended |= step.tokens.iter().any(|token| eos.contains(token));
                    arrivals.push(json!({"elapsed_s":elapsed,"tokens":step.tokens.len()}));
                    tokens.extend(step.tokens);
                }
            }
            assert!(!tokens.is_empty());
            if repetition > 0 && prefix_cache_bytes > 0 {
                assert!(request.prefix.statistics.cached_tokens > 0);
            }
            if prefix_cache_bytes == 0 {
                assert_eq!(request.prefix.statistics.cached_tokens, 0);
            }
            if fixture["mtp_drafts"].as_u64() == Some(0) {
                assert_eq!(request.mtp.rounds, 0);
            }
            if let Some(previous) = deterministic.get(&case) {
                assert_eq!(&tokens, previous, "Cached generation changed seeded output");
            } else {
                deterministic.insert(case, tokens.clone());
            }
            let wall_s = started.elapsed().as_secs_f64();
            if tracing {
                // SAFETY: The timed request completed; stop diagnostic collection.
                unsafe {
                    crate::backend::cuda::check(
                        cudarc::driver::sys::cuProfilerStop(),
                        "stop diagnostic trace",
                    )
                    .unwrap();
                }
            }
            let last = arrivals.last().unwrap()["elapsed_s"].as_f64().unwrap();
            let decode_s = last - first.unwrap();
            let decode_tokens = tokens.len() - first_count;
            // Download after the timed generation. Compare every live cached
            // KV/GDN/conv/PLE/MTP range across arithmetic-preserving packages.
            let state_hashes = (fixture["state_hashes"].as_bool() == Some(true)).then(|| {
                model.execution.sync().unwrap();
                let position = model.read_control(&model.manifest.position).unwrap() as usize;
                crate::runtime::mtp_gpu_tests::complete_prefix_state(&model, position)
            });
            let row = json!({"case":case,"repetition":repetition,"warmup":repetition==0,
                "prompt_tokens":source.input_tokens.len(),"output_tokens":tokens,
                "first_step_tokens":first_count,"decode_tokens":decode_tokens,
                "decode_s":decode_s,"decode_tps":(decode_s>0.).then(||decode_tokens as f64/decode_s),
                "wall_s":wall_s,"ttft_s":first,"arrivals":arrivals,
                "prompt_through_first_token_tps":source.input_tokens.len() as f64/first.unwrap(),
                "prefix":request.prefix.statistics,"mtp":request.mtp,
                "sampling":source.sampling,"state_hashes":state_hashes,
                "finish_reason":if ended {"stop"}else{"length"}});
            eprintln!(
                "cached generation case={case} repetition={repetition}: {} TPS",
                row["decode_tps"]
            );
            rows.push(row);
            model.finish_request(&mut request, true).unwrap();
            std::fs::write(&output,serde_json::to_vec_pretty(&json!({"rows":rows,
                "scope":"Real full autoregressive requests with fixture-selected prefix cache and MTP, normal scheduler. Committed tokens only; decode excludes the complete first emitted step. Prompt throughput includes request setup and first token selection. Warmup excluded; seeded output equality checked."})).unwrap()).unwrap();
        }
    }
}

#[test]
#[ignore = "Requires real model and exclusive GPU experiment lock"]
fn profile_speculative_verification() {
    let fixture: serde_json::Value = crate::model::read(&std::path::PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    let output = std::path::PathBuf::from(fixture["output"].as_str().unwrap());
    assert!(!output.exists());
    let sources: Vec<GenerationInput> = serde_json::from_value(fixture["cases"].clone()).unwrap();
    let widths: Vec<usize> = serde_json::from_value(fixture["widths"].clone()).unwrap();
    let mut model = ModelRuntime::load_with_options(
        std::path::Path::new(fixture["model"].as_str().unwrap()),
        LoadOptions {
            cuda_graph: crate::execution::CudaGraphMode::DecodeOnly,
            prefix_cache_bytes: 0,
            ..LoadOptions::default()
        },
    )
    .unwrap();
    let mut rows = vec![];
    for source in sources {
        for &width in &widths {
            let mut tokens = vec![];
            model
                .generate(
                    &source.input_tokens,
                    None,
                    width + 1,
                    &source.sampling,
                    || false,
                    |t| {
                        tokens.push(t);
                        true
                    },
                )
                .unwrap();
            assert!(tokens.len() >= width);
            let mut request = model
                .start_request(
                    GenerationInput {
                        audios: vec![],
                        input_tokens: source.input_tokens.clone(),
                        images: vec![],
                        max_new_tokens: width + 16,
                        sampling: source.sampling.clone(),
                        prefix_hints: vec![],
                    },
                    &|| false,
                )
                .unwrap();
            while request.prefilling {
                model
                    .advance_requests(&mut [&mut request], &scheduler::Options::default())
                    .unwrap();
            }
            let spec = model.manifest.mtp.clone().unwrap();
            let verify = spec.verification_plan(width).unwrap();
            let position = source.input_tokens.len() as u32;
            model
                .upload_ids(&model.manifest.input, &tokens[..width])
                .unwrap();
            model
                .prepare_verification_inputs(
                    &verify.program,
                    &tokens[..width],
                    &source.input_tokens,
                )
                .unwrap();
            model.upload_segment_controls(width).unwrap();
            // Materialize demand-mapped KV/workspace before capturing the
            // instrumented graph. Timing replays advance recurrent state and
            // are diagnostic only; they do not establish output quality.
            model
                .launch_rows(&verify.program, width, ExecutionPhase::Decode)
                .unwrap();
            let plan = row_invocations(&model, &verify.program, width).unwrap();
            let trials = model
                .execution
                .profile_graph_operations_at_position(
                    &plan,
                    Some((&model.manifest.position, position)),
                )
                .unwrap();
            let operations: Vec<_> = plan
                .iter()
                .enumerate()
                .map(|(index, op)| {
                    let name = match &op.operation {
                        crate::model::Operation::Kernel { name } => name.clone(),
                        _ => category(&model, op),
                    };
                    let ms = median(
                        &trials
                            .iter()
                            .map(|t| f64::from(t[index]))
                            .collect::<Vec<_>>(),
                    );
                    json!({"name":name,"median_ms":ms})
                })
                .collect();
            rows.push(json!({"prompt_tokens":source.input_tokens.len(),"verification_tokens":width,
                "program":verify.program,"operations":operations,
                "instrumented_gpu_median_ms":median(&trials.iter().map(|t|t.iter().map(|x|f64::from(*x)).sum()).collect::<Vec<_>>())}));
            std::fs::write(&output,serde_json::to_vec_pretty(&json!({"rows":rows,
                "scope":"Real model, teacher-forced target tokens; instrumented verification graph at fixed position. Recurrent state advances between replays. Diagnostic only, excludes draft/commit/refresh and is not end-to-end TPS or quality acceptance."})).unwrap()).unwrap();
            model.finish_request(&mut request, false).unwrap();
        }
    }
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
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
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
            ..LoadOptions::default()
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
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
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
            ..LoadOptions::default()
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
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
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
            ..LoadOptions::default()
        },
    )
    .unwrap();
    model.manifest.mtp = None;
    let suffixes: Option<Vec<usize>> = fixture
        .get("suffix_tokens")
        .map(|value| serde_json::from_value(value.clone()).unwrap());
    if let Some(suffixes) = &suffixes {
        assert_eq!(suffixes.len(), sources.len());
        assert_eq!(fixture["stream"].as_bool(), Some(true));
    }
    let mut rows = vec![];
    for (case, source) in sources.into_iter().enumerate() {
        let length = source.input_tokens.len();
        let requested = suffixes.as_ref().map_or(length, |sizes| sizes[case]);
        assert!((1..=length).contains(&requested));
        let (chunk, program, head) = model.manifest.select_prefill_plan(requested).unwrap();
        let (program, head) = (program.to_owned(), head.to_owned());
        if suffixes.is_some() {
            assert_eq!(chunk, requested);
        }
        let base = length - chunk;
        // Materialize the exact context and its KV/workspace before profiling.
        // Replays reset only the target position; they are timing diagnostics,
        // not a comparison of the recurrent state or generated text.
        let mut request = model
            .start_request(
                GenerationInput {
                    input_tokens: source.input_tokens.clone(),
                    images: source.images.clone(),
                    audios: source.audios.clone(),
                    max_new_tokens: 1,
                    sampling: source.sampling.clone(),
                    prefix_hints: vec![],
                },
                &|| false,
            )
            .unwrap();
        if suffixes.is_some() {
            while request.offset < base {
                let n = model
                    .manifest
                    .select_prefill_plan(base - request.offset)
                    .unwrap()
                    .0;
                model
                    .with_request(&mut request, |m, r| {
                        m.advance_prefill_step(r, n, false, f64::INFINITY)
                    })
                    .unwrap();
            }
        } else {
            while request.prefilling {
                model
                    .advance_requests(&mut [&mut request], &scheduler::Options::default())
                    .unwrap();
            }
        }
        model.execution.activate_sequence(request.slot).unwrap();
        model
            .upload_ids(&model.manifest.input, &source.input_tokens[base..])
            .unwrap();
        model
            .prepare_prefill_inputs(
                &program,
                &source.input_tokens[base..],
                &source.input_tokens[..base],
            )
            .unwrap();
        model
            .upload_ids(&model.manifest.position, &[base as u32])
            .unwrap();
        model
            .execution
            .ensure_sequence_program(request.slot, &program)
            .unwrap();
        let mut plan = row_invocations(&model, &program, chunk).unwrap();
        if suffixes.is_some() {
            model.upload_segment_controls(chunk).unwrap();
            plan.extend(row_invocations(&model, &head, chunk).unwrap());
        }
        if fixture["ncu"].as_bool() == Some(true) {
            profile_native_ncu(&mut model, &plan);
            rows.push(json!({"input_tokens":length,"operations":plan}));
            std::fs::write(
                &output,
                serde_json::to_vec_pretty(&json!({"rows":rows})).unwrap(),
            )
            .unwrap();
            model.finish_request(&mut request, false).unwrap();
            continue;
        }
        let stream = fixture["stream"].as_bool() == Some(true);
        let trials = if stream {
            model.execution.profile_stream_operations_at_position(
                &plan,
                (&model.manifest.position, base as u32),
                suffixes.is_some(),
            )
        } else {
            model.execution.profile_graph_operations_at_position(
                &plan,
                Some((&model.manifest.position, base as u32)),
            )
        }
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
            "program":program,"operation_count":plan.len(),"launch_mode":if stream { "stream" } else { "graph" },
            "operations":plan,
            "operation_median_ms":(0..plan.len()).map(|i|median(&trials.iter().map(|t|f64::from(t[i])).collect::<Vec<_>>())).collect::<Vec<_>>(),
            "instrumented_gpu_median_ms":median(&trials.iter().map(|t|t.iter().map(|x|f64::from(*x)).sum()).collect::<Vec<f64>>()),
            "gpu_group_median_ms":groups.iter().map(|(k,v)|(k.clone(),median(v))).collect::<BTreeMap<_,_>>() }));
        let scope = if suffixes.is_some() {
            "Real prefix followed by one instrumented incremental target pass, including the head. Stream events include launch gaps. Request setup and MTP warm are outside this measurement."
        } else {
            "Last real-model prefill chunk at fixed target position; three instrumented trials after warm-up; event boundaries include any stream launch gaps in stream mode; head excluded. Recurrent state advances between replays. Diagnostic only, not end-to-end TPS or quality acceptance."
        };
        std::fs::write(
            &output,
            serde_json::to_vec_pretty(
                &json!({"rows":rows,"fingerprint":model.stats.manifest_sha256,
            "scope":scope}),
            )
            .unwrap(),
        )
        .unwrap();
        model.finish_request(&mut request, false).unwrap();
    }
}

#[test]
#[ignore = "Captures real FFN inputs; requires model and exclusive GPU experiment lock"]
fn capture_decode_projections() {
    use std::io::Write;
    let fixture: serde_json::Value = crate::model::read(&std::path::PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
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
            ..LoadOptions::default()
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
                    }], false, false)?;
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
#[ignore = "Real vision encoder plan; requires images and exclusive GPU experiment lock"]
fn profile_vision_stages() {
    let fixture: serde_json::Value = crate::model::read(&std::path::PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
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
            mtp_drafts: Some(0),
            ..LoadOptions::default()
        },
    )
    .unwrap();
    let vision = model
        .manifest
        .vision
        .clone()
        .expect("Vision model required");
    let mut seen = BTreeSet::new();
    let mut rows = vec![];
    for (case, source) in sources.into_iter().enumerate() {
        for (index, image) in source.images.into_iter().enumerate() {
            if !seen.insert((image.grid_height, image.grid_width)) {
                continue;
            }
            let patches = image.grid_height * image.grid_width;
            let tokens = std::iter::once(vision.vision_start_token_id)
                .chain(std::iter::repeat_n(vision.image_token_id, patches / 4))
                .chain([vision.vision_end_token_id])
                .collect::<Vec<_>>();
            // Materialize real pixels, grid, length and buffers before timing.
            // Every replay starts at patch embedding and replaces its outputs.
            model.execution.prepare_reference_sequence().unwrap();
            model.prepare_visual(&tokens, &[image], &|| false).unwrap();
            let program = vision
                .plans
                .iter()
                .filter(|p| p.patches >= patches && p.min_patches <= patches)
                .min_by_key(|p| p.patches)
                .expect("Image must fit a native vision plan")
                .program
                .clone();
            let plan = row_invocations(&model, &program, patches).unwrap();
            if fixture["ncu"].as_bool() == Some(true) {
                profile_native_ncu(&mut model, &plan);
                rows.push(json!({"case":case,"image":index,"patches":patches,
                    "program":program,"operations":plan}));
            } else {
                let hot = model.execution.profile_invocations_hot(&plan, 8).unwrap();
                let trials = model.execution.profile_graph_operations(&plan).unwrap();
                rows.push(json!({"case":case,"image":index,"patches":patches,
                    "program":program,"operations":plan,
                    "hot_graph_median_ms":median(&hot[1..].iter().map(|&v|f64::from(v)).collect::<Vec<_>>()),
                    "instrumented_gpu_median_ms":median(&trials.iter().map(|t|t.iter().map(|x|f64::from(*x)).sum()).collect::<Vec<_>>()),
                    "operation_median_ms":(0..plan.len()).map(|i|median(&trials.iter().map(|t|f64::from(t[i])).collect::<Vec<_>>())).collect::<Vec<_>>() }));
            }
            std::fs::write(&output, serde_json::to_vec_pretty(&json!({"rows":rows,
                "fingerprint":model.stats.manifest_sha256,
                "scope":"Real normalized image patches through the complete native vision encoder. Instrumented GPU graph, excluding image preprocessing, feature copies and text generation; not API latency or quality acceptance."})).unwrap()).unwrap();
        }
    }
    assert!(!rows.is_empty(), "Fixture requires at least one image");
}

#[test]
#[ignore = "Real joint audio/prefill stages; requires exclusive GPU experiment lock"]
fn profile_joint_prefill_stages() {
    let fixture: serde_json::Value = crate::model::read(&std::path::PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
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
            mtp_drafts: Some(0),
            ..LoadOptions::default()
        },
    )
    .unwrap();
    let mut requests: Vec<_> = sources
        .into_iter()
        .map(|source| model.start_request(source, &|| false).unwrap())
        .collect();
    let began = Instant::now();
    model
        .prepare_audio_cohorts(&mut requests.iter_mut().collect::<Vec<_>>())
        .unwrap();
    let audio_ms = began.elapsed().as_secs_f64() * 1000.;
    let capacity = model.manifest.batch_layout.as_ref().unwrap().segment_tokens;
    let mut remaining = capacity;
    let selected: Vec<_> = requests
        .iter()
        .enumerate()
        .filter_map(|(i, r)| {
            let count = remaining.min(r.input.len());
            remaining -= count;
            (count > 0).then_some((i, count))
        })
        .collect();
    let began = Instant::now();
    let segments = model
        .prepare_batch_inputs(&mut requests.iter_mut().collect::<Vec<_>>(), &selected)
        .unwrap();
    let prepare_ms = began.elapsed().as_secs_f64() * 1000.;
    // Repeat the same causal positions. Production packets advance each owner
    // by its segment length; retaining that commit in a microbenchmark would
    // change the context and eventually exceed the prepared KV mappings.
    let mut plan = model.batch_plans.batch_plan(&segments).unwrap();
    plan.retain(|call| {
        !matches!(&call.operation,
        crate::model::Operation::Kernel { name } if name == "qwen_rows/end/advance")
    });
    let mut result = json!({"audio_ms":audio_ms,"prepare_ms":prepare_ms,
        "segments":segments.iter().map(|s|json!({"slot":s.slot,"tokens":s.tokens})).collect::<Vec<_>>(),"operations":plan,
        "scope":"Real uncached recordings encoded as a cohort, followed by a native joint prefill packet. Position commits are excluded so replays overwrite the same causal positions; diagnostics are discarded."});
    if fixture["ncu"].as_bool() == Some(true) {
        profile_native_ncu(&mut model, &plan);
    } else {
        let hot = model.execution.profile_invocations_hot(&plan, 8).unwrap();
        // Bound external-event nodes per diagnostic graph; the Jetson driver
        // cannot replay thousands of external events in one graph reliably.
        let mut trials = vec![vec![]; 3];
        for chunk in plan.chunks(256) {
            let times = model.execution.profile_graph_operations(chunk).unwrap();
            for (trial, values) in trials.iter_mut().zip(times) {
                trial.extend(values);
            }
        }
        result["hot_graph_median_ms"] = json!(median(
            &hot[1..].iter().map(|&v| f64::from(v)).collect::<Vec<_>>()
        ));
        result["operation_median_ms"] = json!(
            (0..plan.len())
                .map(|i| median(&trials.iter().map(|t| f64::from(t[i])).collect::<Vec<_>>()))
                .collect::<Vec<_>>()
        );
    }
    for request in &mut requests {
        model.finish_request(request, false).unwrap();
    }
    std::fs::write(output, serde_json::to_vec_pretty(&result).unwrap()).unwrap();
}

#[test]
#[ignore = "Real audio encoder stages; requires exclusive GPU experiment lock"]
fn profile_audio_stages() {
    let fixture: serde_json::Value = crate::model::read(&std::path::PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
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
            mtp_drafts: Some(0),
            ..LoadOptions::default()
        },
    )
    .unwrap();
    let a = model.manifest.audio.clone().unwrap();
    let mut rows = vec![];
    for (case, source) in sources.iter().enumerate() {
        let input = &source.audios[0];
        let features = input.features().unwrap();
        let started = Instant::now();
        model.prepare_audio_stem(input, &|| false).unwrap();
        let stem_ms = started.elapsed().as_secs_f64() * 1000.;
        // The sequence feature buffer is unused in this encoder diagnostic.
        // Keep the real CNN output there, and restore it before every replay.
        let bytes = features * 1024 * 2;
        model
            .execution
            .copy_range(&a.encoder_input, &a.features, 0, bytes)
            .unwrap();
        let lengths: Vec<u32> = if a.window_size == 0 {
            vec![features as u32]
        } else {
            (0..features)
                .step_by(a.window_size)
                .map(|i| (features - i).min(a.window_size) as u32)
                .collect()
        };
        model.upload_ids(&a.length, &lengths).unwrap();
        for (program, count) in [
            ("audio_stem", input.frames.div_ceil(100).min(a.stem_chunks)),
            ("audio_encoder", features),
        ] {
            let mut plan = row_invocations(&model, program, count).unwrap();
            if program == "audio_encoder" {
                plan.insert(
                    0,
                    Invocation {
                        operation: crate::model::Operation::Copy {
                            source: a.features.clone(),
                            destination: a.encoder_input.clone(),
                            bytes,
                        },
                        sequence: None,
                        launch: None,
                        views: BTreeMap::new(),
                    },
                );
            }
            if fixture["ncu"].as_bool() == Some(true) {
                profile_native_ncu(&mut model, &plan);
            } else {
                let hot = model.execution.profile_invocations_hot(&plan, 8).unwrap();
                let trials = model.execution.profile_graph_operations(&plan).unwrap();
                rows.push(json!({"case":case,"frames":input.frames,"features":features,"program":program,"count":count,"stem_wall_ms":stem_ms,
                    "hot_graph_median_ms":median(&hot[1..].iter().map(|&v|f64::from(v)).collect::<Vec<_>>()),
                    "operations":plan,"operation_median_ms":(0..plan.len()).map(|i|median(&trials.iter().map(|t|f64::from(t[i])).collect::<Vec<_>>())).collect::<Vec<_>>() }));
            }
        }
        std::fs::write(&output, serde_json::to_vec_pretty(&json!({"rows":rows,"scope":"CUDA graph diagnostic with real audio features and reset encoder input; excludes frontend and text generation."})).unwrap()).unwrap();
    }
}

#[test]
#[ignore = "Full native decode plan; requires real model and exclusive GPU experiment lock"]
fn profile_decode_chain() {
    let fixture: serde_json::Value = crate::model::read(&std::path::PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    let output = std::path::PathBuf::from(fixture["output"].as_str().unwrap());
    assert!(!output.exists());
    let mut source: GenerationInput = serde_json::from_value(fixture["cases"][0].clone()).unwrap();
    source.max_new_tokens = 256;
    let mut model = ModelRuntime::load_with_options(
        std::path::Path::new(fixture["model"].as_str().unwrap()),
        LoadOptions {
            cuda_graph: crate::execution::CudaGraphMode::DecodeOnly,
            prefix_cache_bytes: 0,
            mtp_drafts: Some(0),
            ..LoadOptions::default()
        },
    )
    .unwrap();
    let mut request = model.start_request(source, &|| false).unwrap();
    let options = scheduler::Options::default();
    while request.prefilling {
        model
            .advance_requests(&mut [&mut request], &options)
            .unwrap();
    }
    let mut wall_ms = vec![];
    let mut tokens = vec![];
    for iteration in 0..24 {
        let started = Instant::now();
        let steps = model
            .advance_requests(&mut [&mut request], &options)
            .unwrap();
        let elapsed = started.elapsed().as_secs_f64() * 1000.;
        assert_eq!(steps.iter().map(|s| s.tokens.len()).sum::<usize>(), 1);
        tokens.extend(steps.into_iter().flat_map(|s| s.tokens));
        if iteration >= 8 {
            wall_ms.push(elapsed);
        }
    }
    let segments = model
        .prepare_batch_inputs(&mut [&mut request], &[(0, 1)])
        .unwrap();
    let plan = model.batch_plans.batch_plan(&segments).unwrap();
    model.execution.sync().unwrap();
    let mut result = json!({
        "model": fixture["model"], "fingerprint": model.stats.manifest_sha256,
        "wall_median_ms": median(&wall_ms),
        "decode_tps": 1000. * wall_ms.len() as f64 / wall_ms.iter().sum::<f64>(),
        "tokens": tokens, "operations": plan,
        "scope": "Normal scheduler, MTP off, 8 warm and 16 measured autoregressive steps. GPU diagnostics below replay the actual native batch plan; their changed states are discarded. Per-operation timing and NCU instrumentation are not request performance."
    });
    if fixture["ncu"].as_bool() == Some(true) {
        profile_native_ncu(&mut model, &plan);
    } else {
        let times = model.execution.profile_invocations_hot(&plan, 8).unwrap();
        result["hot_graph_median_ms"] = json!(median(
            &times[1..].iter().map(|&v| f64::from(v)).collect::<Vec<_>>()
        ));
        let trials = model.execution.profile_graph_operations(&plan).unwrap();
        result["operation_median_ms"] = json!(
            (0..plan.len())
                .map(|i| median(&trials.iter().map(|t| f64::from(t[i])).collect::<Vec<_>>()))
                .collect::<Vec<_>>()
        );
    }
    model.finish_request(&mut request, false).unwrap();
    std::fs::write(output, serde_json::to_vec_pretty(&result).unwrap()).unwrap();
}

#[test]
#[ignore = "Requires real model and exclusive GPU experiment lock"]
fn profile_continuous_decode() {
    let fixture: serde_json::Value = crate::model::read(&std::path::PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
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
            ..LoadOptions::default()
        },
    )
    .unwrap();
    // Isolate target decode. This is not a quality assessment or API benchmark.
    model.manifest.mtp = None;
    if fixture["sampling_batch"].as_bool() == Some(false)
        && let Some(sampling) = &mut model.manifest.sampling
    {
        sampling.batch = None;
    }
    let options = scheduler::Options::default();
    let iterations = fixture["iterations"].as_u64().unwrap_or(64) as usize;
    assert!((1..=128).contains(&iterations));
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
                            audios: source.audios.clone(),
                            input_tokens: source.input_tokens.clone(),
                            images: source.images.clone(),
                            max_new_tokens: 160,
                            sampling: source.sampling.clone(),
                            prefix_hints: vec![],
                        },
                        &|| false,
                    )
                    .unwrap()
            })
            .collect();
        // Prepare the cohort without generating ahead on its earliest rows.
        // Admission/prefill are outside this fixed-batch decode diagnostic.
        for request in &mut requests {
            while request.prefilling {
                model.advance_requests(&mut [request], &options).unwrap();
            }
        }
        for _ in 0..2 {
            model
                .advance_requests(&mut requests.iter_mut().collect::<Vec<_>>(), &options)
                .unwrap();
        }
        let mut walls = vec![];
        let mut phases: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        let live_events = model.execution.profile_events(2).unwrap();
        model.execution.graph_profile.millis.borrow_mut().clear();
        if count == 1 {
            model
                .execution
                .graph_profile
                .events
                .set(Some([live_events[0], live_events[1]]));
        }
        for _ in 0..iterations {
            MARKS.with(|m| *m.borrow_mut() = Some(vec![]));
            let started = Instant::now();
            let steps = model
                .advance_requests(&mut requests.iter_mut().collect::<Vec<_>>(), &options)
                .unwrap();
            mark("iteration_end");
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
        model.execution.graph_profile.events.set(None);
        if fixture["live_only"].as_bool() == Some(true) {
            let gpu: Vec<_> = model
                .execution
                .graph_profile
                .millis
                .borrow()
                .iter()
                .copied()
                .map(f64::from)
                .collect();
            assert_eq!(gpu.len(), walls.len());
            rows.push(json!({"batch":count,"iterations":walls.len(),
                "wall_mean_ms":walls.iter().sum::<f64>()/walls.len() as f64,
                "live_graph_mean_ms":gpu.iter().sum::<f64>()/gpu.len() as f64,
                "phases_mean_ms":phases.iter().map(|(k,v)|(k.clone(),v.iter().sum::<f64>()/v.len() as f64)).collect::<BTreeMap<_,_>>() }));
            model.finish_request(&mut requests[0], false).unwrap();
            continue;
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
                .ensure_sequence_program_at(request.slot, "decode", request.history.len() + 128, 1)
                .unwrap();
        }
        // Reserve can invalidate graph/workspace bindings. Prepare a fresh
        // native packet before replaying, including the one-request packet.
        model
            .advance_requests(&mut requests.iter_mut().collect::<Vec<_>>(), &options)
            .unwrap();
        let plan = model.batch_plans.batch_plan(&segments).unwrap();
        if fixture["ncu"].as_bool() == Some(true) {
            profile_native_ncu(&mut model, &plan);
            rows.push(json!({"batch":count,"operations":plan,"wall_median_ms":median(&walls)}));
            for request in &mut requests {
                model.finish_request(request, false).unwrap();
            }
            continue;
        }
        // Sample actual routing outside every timing window. Diverse requests
        // can touch many more expert banks than repeated-prompt diagnostics.
        let mut routing = Vec::new();
        if fixture["routing_statistics"].as_bool() == Some(true) {
            for operation in &plan {
                model
                    .execution
                    .execute_batch(vec![], std::slice::from_ref(operation), false, false)
                    .unwrap();
                if let crate::model::Operation::Kernel { name } = &operation.operation
                    && name.contains("/post/expert_up/")
                {
                    let raw = model
                        .execution
                        .download_bytes("MoeCounts", model.execution.sizes["MoeCounts"])
                        .unwrap();
                    let (words, remainder) = raw.as_chunks::<4>();
                    assert!(remainder.is_empty(), "Misaligned expert count buffer");
                    let counts: Vec<_> = words.iter().map(|b| i32::from_le_bytes(*b)).collect();
                    assert!(counts.iter().all(|&n| n >= 0));
                    routing.push(json!({"kernel":name,"counts":counts}));
                }
            }
            assert!(
                !routing.is_empty(),
                "No expert projections in routing diagnostic"
            );
        }
        let gpu: Vec<f64> = model
            .execution
            .profile_hot_graph(&key, 16)
            .unwrap()
            .into_iter()
            .map(f64::from)
            .collect();
        if fixture["hot_only"].as_bool() == Some(true) {
            rows.push(json!({"batch":count,"iterations":iterations,
                "wall_median_ms":median(&walls),
                "target_tokens_per_second":count as f64*1000./(walls.iter().sum::<f64>()/walls.len() as f64),
                "phases_median_ms":phases.iter().map(|(k,v)|(k.clone(),median(v))).collect::<BTreeMap<_,_>>(),
                "hot_graph_gpu_median_ms":median(&gpu),"operations":plan}));
            for request in &mut requests {
                model.finish_request(request, false).unwrap();
            }
            continue;
        }
        let trials = model.execution.profile_graph_operations(&plan).unwrap();
        let mut groups: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        let mut operations = Vec::with_capacity(plan.len());
        for (index, op) in plan.iter().enumerate() {
            let name = match &op.operation {
                crate::model::Operation::Kernel { name } => name.clone(),
                _ => category(&model, op),
            };
            operations.push(json!({"name":name,"median_ms":median(
                &trials.iter().map(|t|f64::from(t[index])).collect::<Vec<_>>()
            )}));
            let values = groups
                .entry(category(&model, op))
                .or_insert(vec![0.; trials.len()]);
            for (trial, times) in trials.iter().enumerate() {
                values[trial] += f64::from(times[index]);
            }
        }
        let mut subsets = BTreeMap::new();
        if let Some(filters) = fixture["kernel_subsets"].as_object() {
            // Isolated diagnostic graphs have stale inputs. Their times are
            // neither additive nor valid model executions; do this last.
            for (label, names) in filters {
                let names: BTreeSet<String> = serde_json::from_value(names.clone()).unwrap();
                let selected: Vec<_> = plan.iter().filter(|op| matches!(
                    &op.operation, crate::model::Operation::Kernel { name } if names.contains(name)
                )).cloned().collect();
                assert_eq!(selected.len(), names.len(), "Unknown subset kernel");
                assert!(!selected.is_empty());
                let times: Vec<_> = model
                    .execution
                    .profile_invocations_hot(&selected, 8)
                    .unwrap()
                    .into_iter()
                    .skip(1)
                    .map(f64::from)
                    .collect();
                subsets.insert(label.clone(), json!({"operations":selected.len(),
                    "median_ms":median(&times), "scope":"Isolated real kernels with stale inputs and two external timing events; diagnostic only, not additive and not model inference."}));
            }
        }
        let row = json!({"batch":count,"iterations":iterations,"wall_median_ms":median(&walls),
            "target_tokens_per_second":count as f64*1000./(walls.iter().sum::<f64>()/walls.len() as f64),
            "phases_median_ms":phases.iter().map(|(k,v)|(k.clone(),median(v))).collect::<BTreeMap<_,_>>(),
            "hot_graph_gpu_median_ms":median(&gpu),"operation_count":plan.len(),
            "instrumented_gpu_median_ms":median(&trials.iter().map(|t| t.iter().map(|x|f64::from(*x)).sum()).collect::<Vec<f64>>()),
            "gpu_operations":operations,
            "routing_statistics":routing,
            "isolated_subsets":subsets,
            "gpu_group_median_ms":groups.iter().map(|(k,v)|(k.clone(),median(v))).collect::<BTreeMap<_,_>>()});
        eprintln!("PROFILE {row}");
        rows.push(row);
        std::fs::write(&output,serde_json::to_vec_pretty(&json!({"scope":"Real target decode iterations per fixed batch (count recorded in each row); no MTP, admission/prefill/API excluded. GPU operation timing uses external-event nodes in an instrumented replay, compare its overhead with the uninstrumented hot graph.","rows":rows})).unwrap()).unwrap();
        for request in &mut requests {
            model.finish_request(request, false).unwrap();
        }
    }
    if fixture["live_only"].as_bool() == Some(true)
        || fixture["ncu"].as_bool() == Some(true)
        || fixture["hot_only"].as_bool() == Some(true)
    {
        std::fs::write(&output,serde_json::to_vec_pretty(&json!({"scope":"Live autoregressive iterations; two graph timing events, CPU stage timestamps; diagnostic only.","rows":rows})).unwrap()).unwrap();
    }
}

#[test]
#[ignore = "Captures real prefill partition intermediates; requires exclusive GPU lock"]
fn capture_prefill_partitions() {
    let spec: serde_json::Value = crate::model::read(&std::path::PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    let output = std::path::PathBuf::from(spec["output"].as_str().unwrap());
    std::fs::create_dir(&output).unwrap();
    let source: GenerationInput = serde_json::from_value(spec["case"].clone()).unwrap();
    let mut model = ModelRuntime::load_with_options(
        std::path::Path::new(spec["model"].as_str().unwrap()),
        LoadOptions {
            cuda_graph: crate::execution::CudaGraphMode::Off,
            prefix_cache_bytes: 0,
            mtp_drafts: Some(0),
            ..Default::default()
        },
    )
    .unwrap();
    let mut results = vec![];
    for partition in spec["partitions"].as_array().unwrap() {
        let label = partition["name"].as_str().unwrap();
        let chunks: Vec<usize> = serde_json::from_value(partition["chunks"].clone()).unwrap();
        assert_eq!(chunks.iter().sum::<usize>(), source.input_tokens.len());
        let mut req = model
            .start_request(
                GenerationInput {
                    audios: vec![],
                    input_tokens: source.input_tokens.clone(),
                    images: source.images.clone(),
                    max_new_tokens: 1,
                    sampling: source.sampling.clone(),
                    prefix_hints: vec![],
                },
                &|| false,
            )
            .unwrap();
        let directory = output.join(label);
        std::fs::create_dir(&directory).unwrap();
        let mut captures = vec![];
        for (index, &chunk) in chunks.iter().enumerate() {
            if index != 0 && index + 1 != chunks.len() {
                model
                    .with_request(&mut req, |m, r| m.advance_prefill(r, chunk, false))
                    .unwrap();
                continue;
            }
            let phase = if index == 0 { "first" } else { "last" };
            model
                .with_request(&mut req, |m, r| {
                    let (_, program, head) = m.manifest.select_prefill_plan(chunk)?;
                    let program = program.to_owned();
                    let head = m.manifest.prefill_plan(chunk).map(|_| head.to_owned());
                    m.upload_ids(
                        if head.is_none() {
                            &m.manifest.token
                        } else {
                            &m.manifest.input
                        },
                        &r.input[r.offset..r.offset + chunk],
                    )?;
                    m.prepare_prefill_inputs(
                        &program,
                        &r.input[r.offset..r.offset + chunk],
                        &r.input[..r.offset],
                    )?;
                    m.upload_segment_controls(chunk)?;
                    m.execution.ensure_sequence_program(r.slot, &program)?;
                    for invocation in row_invocations(m, &program, chunk)? {
                        let name = match &invocation.operation {
                            crate::model::Operation::Kernel { name } => Some(name.clone()),
                            _ => None,
                        };
                        m.execution
                            .execute_batch(vec![], &[invocation], false, false)?;
                        if let Some(watches) = name
                            .as_ref()
                            .and_then(|n| spec["watches"].get(n))
                            .and_then(|v| v.as_array())
                        {
                            for watch in watches {
                                let buffer = watch["buffer"].as_str().unwrap();
                                let offset = watch["offset"].as_u64().unwrap_or(0) as usize;
                                let bytes = watch["bytes"].as_u64().unwrap() as usize;
                                let raw = m.execution.download_bytes(buffer, offset + bytes)?;
                                let file =
                                    format!("{phase}-{}.bin", watch["name"].as_str().unwrap());
                                std::fs::write(directory.join(&file), &raw[offset..])
                                    .map_err(|e| e.to_string())?;
                                captures.push(
                                    json!({"phase":phase,"kernel":name,"watch":watch,"file":file}),
                                );
                            }
                        }
                    }
                    if r.offset + chunk == r.input.len()
                        && let Some(head) = head
                    {
                        m.launch_rows(&head, chunk, ExecutionPhase::Prefill)?;
                    }
                    m.after_prefill(r, chunk)?;
                    Ok(())
                })
                .unwrap();
        }
        let raw = model
            .with_request(&mut req, |m, _| {
                m.execution
                    .download_bytes(&m.manifest.logits, m.manifest.vocab * 4)
            })
            .unwrap();
        std::fs::write(directory.join("logits.f32"), raw).unwrap();
        let state = if spec["capture_state"].as_bool() == Some(true) {
            model
                .with_request(&mut req, |m, _| {
                    m.prefix_ranges(source.input_tokens.len())?
                        .into_iter()
                        .map(|(name, range)| {
                            let raw = m
                                .execution
                                .download_bytes(&name, range.offset + range.bytes)?;
                            Ok((name, crate::artifact::sha256(&raw[range.offset..])))
                        })
                        .collect::<Result<BTreeMap<_, _>>>()
                })
                .unwrap()
        } else {
            BTreeMap::new()
        };
        model.finish_request(&mut req, false).unwrap();
        results.push(json!({"name":label,"chunks":chunks,"captures":captures,"state":state}));
        std::fs::write(
            output.join("capture.json"),
            serde_json::to_vec_pretty(&json!({"cases":results,"seed":20261002})).unwrap(),
        )
        .unwrap();
        eprintln!("prefill capture complete: {label}");
    }
}

#[test]
#[ignore = "Requires real model and exclusive GPU experiment lock"]
fn profile_adaptive_mtp_decode() {
    #[derive(serde::Deserialize)]
    struct Fixture {
        model: std::path::PathBuf,
        output: std::path::PathBuf,
        cases: Vec<GenerationInput>,
        batches: Vec<usize>,
        iterations: usize,
        repetitions: usize,
        mtp_drafts: usize,
        #[serde(default)]
        force_mtp: bool,
    }
    let fixture: Fixture = crate::model::read(&std::path::PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    assert!(!fixture.output.exists());
    assert!(!fixture.cases.is_empty());
    assert!((1..=64).contains(&fixture.iterations) && fixture.repetitions > 0);
    assert!(fixture.batches.iter().all(|b| (1..=128).contains(b)));
    let mut model = ModelRuntime::load_with_options(
        &fixture.model,
        LoadOptions {
            cuda_graph: crate::execution::CudaGraphMode::DecodeOnly,
            prefix_cache_bytes: 0,
            mtp_drafts: Some(fixture.mtp_drafts),
            ..LoadOptions::default()
        },
    )
    .unwrap();
    let mtp = model.manifest.mtp.clone().unwrap();
    let options = scheduler::Options {
        max_active: 128,
        ..Default::default()
    };
    let totals = |requests: &[RequestState]| -> [usize; 3] {
        requests.iter().fold([0; 3], |mut sum, req| {
            sum[0] += req.mtp.rounds;
            sum[1] += req.mtp.proposed_tokens;
            sum[2] += req.mtp.accepted_draft_tokens;
            sum
        })
    };
    let mut rows = vec![];
    for count in fixture.batches {
        for repetition in 0..fixture.repetitions {
            // Alternate paired order; both modes use the normal scheduler and
            // identical distinct prompt histories. Neither forces speculation.
            for enabled in if repetition.is_multiple_of(2) {
                [false, true]
            } else {
                [true, false]
            } {
                model.manifest.mtp = enabled.then(|| mtp.clone());
                model.dynamic_speculation = Default::default();
                if enabled && fixture.force_mtp {
                    model.dynamic_speculation.fixed_width = Some(fixture.mtp_drafts + 1);
                }
                let mut requests: Vec<_> = (0..count)
                    .map(|i| {
                        let source = &fixture.cases[i % fixture.cases.len()];
                        assert!(source.images.is_empty());
                        model
                            .start_request(
                                GenerationInput {
                                    audios: vec![],
                                    input_tokens: source.input_tokens.clone(),
                                    images: vec![],
                                    max_new_tokens: 512,
                                    sampling: source.sampling.clone(),
                                    prefix_hints: vec![],
                                },
                                &|| false,
                            )
                            .unwrap()
                    })
                    .collect();
                for request in &mut requests {
                    while request.prefilling {
                        model.advance_requests(&mut [request], &options).unwrap();
                    }
                }
                // Warm graph capture, ordinary cost estimation and initial MTP
                // probes before measuring. Retain the learned policy normally.
                for _ in 0..8 {
                    model
                        .advance_requests(&mut requests.iter_mut().collect::<Vec<_>>(), &options)
                        .unwrap();
                }
                let before = totals(&requests);
                let before_depths = model.scheduler_statistics.mtp_depth_histogram.clone();
                let mut emitted = 0;
                let began = Instant::now();
                for _ in 0..fixture.iterations {
                    assert!(requests.iter().all(|r| !r.is_finished()));
                    let steps = model
                        .advance_requests(&mut requests.iter_mut().collect::<Vec<_>>(), &options)
                        .unwrap();
                    emitted += steps.iter().map(|s| s.tokens.len()).sum::<usize>();
                }
                let seconds = began.elapsed().as_secs_f64();
                let after = totals(&requests);
                let proposed = after[1] - before[1];
                let accepted = after[2] - before[2];
                if !enabled {
                    assert_eq!(after, [0; 3]);
                    assert_eq!(emitted, count * fixture.iterations);
                }
                let depth_histogram: BTreeMap<_, _> = model
                    .scheduler_statistics
                    .mtp_depth_histogram
                    .iter()
                    .map(|(&depth, &count)| {
                        (
                            depth,
                            count - before_depths.get(&depth).copied().unwrap_or(0),
                        )
                    })
                    .collect();
                if enabled && fixture.force_mtp {
                    assert!(proposed > 0, "Forced MTP never executed");
                }
                let row = json!({"batch":count,"repetition":repetition,"mtp_enabled":enabled,
                    "iterations":fixture.iterations,"committed_tokens":emitted,"wall_s":seconds,
                    "decode_tps":emitted as f64/seconds,"mtp_rounds":after[0]-before[0],
                    "proposed_tokens":proposed,"accepted_draft_tokens":accepted,
                    "depth_histogram":depth_histogram,"forced":fixture.force_mtp,
                    "acceptance":(proposed>0).then(||accepted as f64/proposed as f64)});
                eprintln!("adaptive MTP decode {row}");
                rows.push(row);
                for request in &mut requests {
                    model.finish_request(request, false).unwrap();
                }
                std::fs::write(&fixture.output,serde_json::to_vec_pretty(&json!({"rows":rows,
                    "scheduler":options,"scope":"Fixed active cohort; normal adaptive MTP policy, actual committed tokens only. Eight warmup calls excluded; admission/prefill/API excluded. Distinct frozen translation prompts, EOS intentionally ignored as in the target decode diagnostic; no request reaches its output budget."})).unwrap()).unwrap();
            }
        }
    }
}
