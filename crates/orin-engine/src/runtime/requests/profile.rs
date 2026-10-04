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
                    if op.sequence.is_some() {
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
    for count in [1, 2, 4, 8, 16, 32] {
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
