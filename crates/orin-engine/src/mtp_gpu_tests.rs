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
}

#[derive(Deserialize)]
struct Case {
    id: String,
    input_tokens: Vec<u32>,
    max_new_tokens: usize,
    #[serde(default)]
    stop_after: Option<usize>,
}

fn download(model: &ModelRuntime, name: &str, bytes: usize) -> Vec<u8> {
    assert!(bytes <= model.sizes[name]);
    let mut raw = vec![0u8; bytes];
    // SAFETY: The completed graph owns the source; both extents are checked.
    unsafe {
        check(
            (model.session.driver.download)(raw.as_mut_ptr().cast(), model.pointers[name], bytes),
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
                model.sizes[name]
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
    let mut model = ModelRuntime::load(&fixture.model).unwrap();
    let spec = model.manifest.mtp.clone().expect("Native MTP plan");
    let options = crate::sampling::Options {
        temperature: 0.,
        ..Default::default()
    };
    let mut rows = vec![];
    // Alternating modes and fixtures exercises request isolation. Warmup is
    // excluded from performance summaries. Every call generates real tokens.
    for repetition in 0..=fixture.repetitions {
        for case in &fixture.cases {
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
                        None,
                        case.max_new_tokens,
                        &options,
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
                if let Some(expected) = &reference {
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
                    reference_state = Some(state);
                }
                let decode_s = times.last().unwrap() - times[0];
                let tps = (count > 1).then(|| (count - 1) as f64 / decode_s);
                let row = json!({"case":case.id,"mtp":use_mtp,"repetition":repetition,
                    "warmup":repetition==0,"input_tokens":case.input_tokens.len(),
                    "output_tokens":output,"token_times_s":times,"ttft_s":times[0],
                    "decode_s":decode_s,"decode_tps":tps,"generation_s":full_s,
                    "statistics":model.speculation_statistics,
                    "target_position":position,"target_outputs_and_state_equal":true});
                eprintln!(
                    "{} mtp={use_mtp} repeat={repetition} {count} tokens {tps:?} TPS",
                    case.id
                );
                rows.push(row);
                std::fs::write(&fixture.output, serde_json::to_vec_pretty(&json!({
                    "status":"running","seed":crate::sampling::EVALUATION_SEED,
                    "manifest_sha256":model.stats.manifest_sha256,"rows":rows,
                    "scope":"Real greedy autoregressive generation; committed output tokens only. Decode excludes first token and uses callback arrival times. No teacher-forced output."})).unwrap()).unwrap();
            }
        }
    }
    let mut result: serde_json::Value = crate::model::read(&fixture.output).unwrap();
    result["status"] = json!("passed");
    std::fs::write(&fixture.output, serde_json::to_vec_pretty(&result).unwrap()).unwrap();
}
